import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import _LinearWithBias

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import to_dense_batch

from graphgym.config import cfg
from graphgym.models.layer import MLP, BatchNorm1dNode, BatchNorm1dEdge
from graphgym.models.feature_encoder import node_encoder_dict, edge_encoder_dict
from graphgym.register import register_network


CONV_LOOKUP = {
    "GCN": pyg_nn.GCNConv,
    "GAT": pyg_nn.GATConv,
    "GraphSage": pyg_nn.SAGEConv
}


def conv_lookup(model):
    if model not in CONV_LOOKUP:
        raise ValueError("Model {} unavailable".format(model))
    return CONV_LOOKUP[model]


class GMTLayer(nn.Module):
    
    def __init__(self, dim_embed, num_heads, dropout, **kwargs):
        super(GMTLayer, self).__init__()
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.dim_heads = int(dim_embed / num_heads)
        self.dropout = dropout

        self.layer_norm_h = nn.LayerNorm(dim_embed)
        self.layer_norm_z = nn.LayerNorm(dim_embed)
        self.feed_forward = nn.Linear(dim_embed, dim_embed) # use in row-wise fashion

    def multi_head_attention(self, x_dense, graph=None):
        raise NotImplementedError

    def forward(self, x_dense, graph=None):
        query, attention_output = self.multi_head_attention(x_dense, graph)
        embed = self.layer_norm_h(query + attention_output)
        embed = self.layer_norm_z(embed + self.feed_forward(embed))
        return embed


class GMTConvAttention(GMTLayer):

    def __init__(self, *args, **kwargs):
        super(GMTConvAttention, self).__init__(*args, **kwargs)
        self.num_seeds = kwargs['num_seeds']
        self.model_type = kwargs['model_type']

        self.seed = nn.Parameter(torch.empty(self.num_seeds, 1, self.dim_embed))

        if self.num_seeds > 1:
            conv = conv_lookup(self.model_type)
            self.convs_k = nn.ModuleList([
                conv(self.dim_embed, self.dim_heads) for _ in range(self.num_heads)
            ])
            self.convs_v = nn.ModuleList([
                conv(self.dim_embed, self.dim_heads) for _ in range(self.num_heads)
            ])
        else:
            self.convs_k = nn.ModuleList([nn.Linear(self.dim_embed, self.dim_embed)])
            self.convs_v = nn.ModuleList([nn.Linear(self.dim_embed, self.dim_embed)])

        self.q_proj_weight = nn.Parameter(torch.empty(self.dim_embed, self.dim_embed))
        self.out_proj = _LinearWithBias(self.dim_embed, self.dim_embed)
        
        nn.init.xavier_uniform_(self.seed)
        nn.init.xavier_uniform_(self.q_proj_weight)

    def multi_head_attention(self, x_dense, graph=None):
        query = self.seed.repeat(1, x_dense.shape[0], 1)
        
        if graph is not None:
            x, edge_index, batch = graph
            static_k = torch.cat([conv(x, edge_index) for conv in self.convs_k], dim=1)
            static_v = torch.cat([conv(x, edge_index) for conv in self.convs_v], dim=1)
            static_k, _ = to_dense_batch(static_k, batch)
            static_v, _ = to_dense_batch(static_v, batch)
        else:
            static_k = torch.cat([conv(x_dense) for conv in self.convs_k], dim=1)
            static_v = torch.cat([conv(x_dense) for conv in self.convs_v], dim=1)
        
        # reshape for torch.nn.functional
        shape = (x_dense.shape[0] * self.num_heads, self.dim_embed // self.num_heads, -1)
        static_k = static_k.transpose(1, 2).contiguous().view(*shape).transpose(1, 2)
        static_v = static_v.transpose(1, 2).contiguous().view(*shape).transpose(1, 2)

        # requires modification to torch.nn.functional
        output, weights = F.multi_head_attention_forward(
            query,
            None, # TODO
            None, # TODO
            self.dim_embed,
            self.num_heads,
            in_proj_weight=None, 
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight,
            static_k=static_k,
            static_v=static_v
        )

        return query.transpose(0, 1), output.transpose(0, 1)


class GMTSelfAttention(GMTLayer):

    def __init__(self, *args, **kwargs):
        super(GMTSelfAttention, self).__init__(*args, **kwargs)
        self.MHA = nn.MultiheadAttention(self.dim_embed, self.num_heads, dropout=self.dropout)

    def multi_head_attention(self, x, graph):
        output, weights = self.MHA(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        return x, output.transpose(0, 1)


class GraphMultisetTransformerGNN(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(GraphMultisetTransformerGNN, self).__init__()
        self.dim_embed = cfg.gmt.embed
        self.num_heads = cfg.gmt.heads
        self.num_seeds = cfg.gmt.seeds
        self.num_convs = cfg.gmt.convs
        self.dropout = cfg.gmt.dropout
        self.model_type = cfg.gmt.model_type
        self.output_layers = cfg.gmt.output_layers

        if cfg.dataset.node_encoder:
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            dim_in = cfg.dataset.encoder_dim
        if cfg.dataset.edge_encoder:
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)

        conv = conv_lookup(self.model_type)
        self.convs = nn.ModuleList(
            [conv(dim_in, self.dim_embed)] +
            [conv(self.dim_embed, self.dim_embed) for _ in range(self.num_convs - 1)]
        )

        # pooling
        self.attn_1 = GMTConvAttention(
            self.dim_embed, self.num_heads, self.dropout, num_seeds=self.num_seeds, model_type=self.model_type)
        self.attn_2 = GMTSelfAttention(
            self.dim_embed, self.num_heads, dropout=self.dropout)
        self.attn_3 = GMTConvAttention(
            self.dim_embed, self.num_heads, self.dropout, num_seeds=1, model_type=None)

        self.head = nn.Sequential(
            nn.Linear(self.dim_embed, self.dim_embed),
            nn.Linear(self.dim_embed, dim_out)
        )

    def forward(self, batch):
        batch = self.node_encoder_bn(self.node_encoder(batch))
        batch = self.edge_encoder_bn(self.edge_encoder(batch))
        
        x, edge_index, x_batch = batch.node_feature, batch.edge_index, batch.batch
        
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
                
        # x_dense = pyg_nn.global_add_pool(x, x_batch)

        x_dense, mask = to_dense_batch(x, x_batch)

        x_dense = self.attn_1(x_dense, graph=(x, edge_index, x_batch))
        
        x_dense = self.attn_2(x_dense)
        
        x_dense = self.attn_3(x_dense, graph=None)
        
        batch.graph_feature = self.head(x_dense).squeeze()
    
        return batch.graph_feature, batch.graph_label


register_network('gmt', GraphMultisetTransformerGNN)
