import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import _LinearWithBias

import torch_geometric.nn as pyg_nn

from graphgym.config import cfg
from graphgym.models.layer import MLP, BatchNorm1dNode
from graphgym.models.feature_encoder import node_encoder_dict
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

    def multi_head_attention(self, x, edge_index):
        raise NotImplementedError

    def forward(self, x, edge_index=None):
        query, attention_output = self.multi_head_attention(x, edge_index)
        embed = self.layer_norm_h(query + attention_output)
        embed = self.layer_norm_z(embed + self.feed_forward(embed))
        return embed


class GMTConvAttention(GMTLayer):

    def __init__(self, *args, **kwargs):
        super(GMTConvAttention, self).__init__(*args, **kwargs)
        self.num_seeds = kwargs['num_seeds']
        self.model_type = kwargs['model_type']

        self.seed = nn.Parameter(torch.empty(self.num_seeds, self.dim_embed))

        conv = conv_lookup(self.model_type)
        self.convs_k = nn.ModuleList([
            conv(self.dim_embed, self.dim_heads) for _ in range(self.num_heads)
        ])
        self.convs_v = nn.ModuleList([
            conv(self.dim_embed, self.dim_heads) for _ in range(self.num_heads)
        ])
        
        self.q_proj_weight = nn.Parameter(torch.empty(self.dim_embed, self.dim_embed))
        self.out_proj = _LinearWithBias(self.dim_embed, self.dim_embed)

    def multi_head_attention(self, x, edge_index):
        query = self.seed.unsqueeze(1)
        static_k = torch.cat([conv(x, edge_index).unsqueeze(0) for conv in self.convs_k], dim=0) # (N*num_heads, N, E/num_heads)
        static_v = torch.cat([conv(x, edge_index).unsqueeze(0) for conv in self.convs_v], dim=0) # (N*num_heads, N, E/num_heads)

        output, weights = F.multi_head_attention_forward(
            query,
            None,
            None,
            self.dim_embed,
            self.num_heads,
            None,
            None,
            None,
            None,
            False,
            self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight,
            static_k=static_k,
            static_v=static_v
        )
        
        return self.seed, output.squeeze()


class GMTSelfAttention(GMTLayer):

    def __init__(self, *args, **kwargs):
        super(GMTSelfAttention, self).__init__(*args, **kwargs)
        self.MHA = nn.MultiheadAttention(self.dim_embed, self.num_heads, dropout=self.dropout)

    def multi_head_attention(self, x, edge_index=None):
        output, weights = self.MHA(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return x, output.squeeze()


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
        
        assert cfg.dataset.transform != 'ego'

        if cfg.dataset.node_encoder:
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            dim_in = cfg.dataset.encoder_dim

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
            self.dim_embed, self.num_heads, self.dropout, num_seeds=1, model_type=self.model_type)

        self.head = nn.Linear(self.dim_embed, dim_out)

    def forward(self, batch):
        print(dir(batch))
        batch = self.node_encoder_bn(self.node_encoder(batch))
        print(batch.batch)
        
        x, edge_index, x_batch = batch.node_feature, batch.edge_index, batch.batch

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.attn_1(x, edge_index)
        x = self.attn_2(x)
        x = self.attn_3(x, torch.arange(self.num_seeds).long().unsqueeze(0).repeat(2, 1).to(x.device))
        
        batch.node_feature = x
        batch.graph_feature = torch.sigmoid(self.head(x))
        
        return batch.graph_feature, batch.graph_label


register_network('gmt', GraphMultisetTransformerGNN)
