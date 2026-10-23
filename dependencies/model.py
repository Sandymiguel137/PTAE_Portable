"""Denoising transformer autoencoder with an AC measurement decoder.

Learned sensor encoding -> compressed attention tokens -> all-node voltage
reconstruction. The physical decoder is Runtime.measurement (Ybus AC map).
No pretrained expert, attack label, test observation, or test target is used.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

SPEC = {
    'version': 'physics-transformer-ae-v1', 'width': 128, 'heads': 4,
    'latent_tokens': 48, 'transformer_layers': 4, 'dropout': 0.1,
    'measurement_reconstruction_weight': 0.05, 'physics_weight': 0.001,
    'physics_penalty': 'mean log1p squared normalized AC reconstruction residual',
    'training_sensor_mask_probability': 0.1,
    'state_target': 'original normalized real/imaginary voltage',
    'reconstruction_target': 'clean AC measurements of original voltage',
    'selection': 'validation state MSE', 'training_seed': 42,
    'epochs': 200, 'patience': 30, 'batch_size': 16,
}


class CrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4*width, width))

    def forward(self, query, context, context_reliability=None):
        normalized = self.context_norm(context)
        attention_bias = None
        if context_reliability is not None:
            # Log reliability survives LayerNorm; a finite floor handles all-missing input.
            attention_bias = context_reliability.clamp_min(1e-4).log()
            attention_bias = attention_bias[:, None, None, :].expand(
                -1, self.attention.num_heads, query.shape[1], -1)
            attention_bias = attention_bias.reshape(-1, query.shape[1], context.shape[1])
        update, _ = self.attention(self.query_norm(query), normalized, normalized,
                                   attn_mask=attention_bias, need_weights=False)
        query = query + update
        return query + self.ff(self.ff_norm(query))


class PhysicsTransformerAE(nn.Module):
    temporal = False

    def __init__(self, ybus, sensors, width=128, heads=4, tokens=48, layers=4, dropout=.1):
        super().__init__()
        nodes = ybus.shape[0]
        self.register_buffer('sensors', torch.as_tensor(sensors, dtype=torch.long))
        # Physical coupling strengths, normalized by destination row.
        adjacency = abs(ybus).astype(np.float32).tocsr()
        adjacency.setdiag(0); adjacency.eliminate_zeros()
        adjacency = adjacency + sp.eye(nodes, dtype=np.float32)
        adjacency = (sp.diags(1/np.asarray(adjacency.sum(1)).ravel()) @ adjacency).tocoo()
        self.register_buffer('physical_adjacency', torch.sparse_coo_tensor(
            np.stack((adjacency.row, adjacency.col)), adjacency.data, (nodes, nodes)).coalesce())
        self.node_embedding = nn.Parameter(torch.randn(nodes, width)*.02)
        self.encoder = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.local_encoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.latents = nn.Parameter(torch.randn(tokens, width)*.02)
        self.compress = CrossAttention(width, heads, dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, heads, 4*width, dropout, activation='gelu', batch_first=True,
            norm_first=True), layers, enable_nested_tensor=False)
        self.expand = CrossAttention(width, heads, dropout)
        self.local_decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.voltage_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        # Global reconstruction supplies node-specific detail beyond shared local decoding.
        self.global_voltage = nn.Sequential(nn.LayerNorm(tokens*width), nn.Linear(tokens*width, 256),
                                            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, nodes*2))
        self.measurement_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.normal_(self.voltage_head[-1].weight, std=.001)
        nn.init.zeros_(self.voltage_head[-1].bias)
        nn.init.normal_(self.global_voltage[-1].weight, std=.001)
        nn.init.zeros_(self.global_voltage[-1].bias)

    def propagate(self, values):
        batch, nodes, width = values.shape
        flat = values.transpose(0, 1).reshape(nodes, batch*width)
        return torch.sparse.mm(self.physical_adjacency, flat).reshape(nodes, batch, width).transpose(0, 1)

    def reconstruct(self, x, reliability=None):
        # Missing values cannot leak through even when passed nonzero placeholders.
        x = torch.cat((x[..., :3]*x[..., 3:4], x[..., 3:]), -1)
        if reliability is not None:
            node_reliability = x.new_zeros(x.shape[:2])
            node_reliability[:, self.sensors] = reliability
            x = torch.cat((x[..., :3]*node_reliability[..., None], x[..., 3:]), -1)
        h = self.encoder(x) + self.node_embedding[None]
        h = h + self.local_encoder(self.propagate(h))
        latent = self.compress(self.latents[None].expand(len(x), -1, -1), h[:, self.sensors], reliability)
        latent = self.transformer(latent)
        decoded = self.expand(h, latent)
        decoded = decoded + self.local_decoder(self.propagate(decoded))
        voltage = self.voltage_head(decoded) + self.global_voltage(latent.flatten(1)).reshape(len(x), -1, 2)
        return voltage, self.measurement_head(decoded[:, self.sensors])

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.reconstruct(x)[0]


def training_loss(model, runtime, raw, available, truth, training):
    # Synthetic missingness only in fitting, with installation indicators retained.
    if training:
        available = available * (torch.rand_like(available) >= SPEC['training_sensor_mask_probability'])
    voltage_ri, reconstructed = model.reconstruct(runtime.input(raw, available))
    state = F.mse_loss(voltage_ri, runtime.target(truth))
    with torch.no_grad():
        clean = runtime.measurement(truth)
        scale = runtime.astd.clamp_min(runtime.cfg.measurement_std_floor)
        clean_normalized = ((clean-runtime.amean)/scale).to(reconstructed.dtype)
    reconstruction = F.smooth_l1_loss(reconstructed, clean_normalized)
    # Differentiable AC decoder supervises physically reconstructed clean observations,
    # never fitting an attacked measurement as if it were clean.
    physical_residual = (runtime.measurement(runtime.decode(voltage_ri))-clean)/scale
    physical = torch.log1p(physical_residual.square()).mean()
    loss = state + SPEC['measurement_reconstruction_weight']*reconstruction + SPEC['physics_weight']*physical
    return loss, state, reconstruction, physical
