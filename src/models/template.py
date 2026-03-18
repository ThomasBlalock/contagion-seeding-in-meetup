# Boilerplate code - fix as needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv # Example layer that supports edge_attr

class MultiplexSeedingModel(nn.Module):
    def __init__(self, node_dim=384, edge_dim=1, event_dim=384, hidden_dim=128):
        super().__init__()
        
        # Encoders
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.event_encoder = nn.Linear(event_dim, hidden_dim)
        
        # Parallel Message Passing Layers (Graph 1 and Graph 2)
        # GATConv accepts edge_attr to condition the attention weights
        self.conv1_1 = GATConv(hidden_dim, hidden_dim // 2, edge_dim=edge_dim)
        self.conv2_1 = GATConv(hidden_dim, hidden_dim // 2, edge_dim=edge_dim)
        
        # Crossflow/Fusion Layer
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Final Scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features, edge_index_1, edge_attr_1, edge_index_2, edge_attr_2, event_features, seed_mask):
        N = node_features.size(0)
        
        # 1. Initial Node Encoding
        x = F.relu(self.node_encoder(node_features))
        
        # 2. Parallel Message Passing
        # Path A: 1-Simplices
        x_1 = F.elu(self.conv1_1(x, edge_index_1, edge_attr_1))
        # Path B: 2-Simplices
        x_2 = F.elu(self.conv2_1(x, edge_index_2, edge_attr_2))
        
        # 3. Crossflow / Fused Representation
        # Concatenating the embeddings from both graph types
        x_fused = torch.cat([x_1, x_2], dim=-1) # Shape: (N, hidden_dim)
        x_nodes = F.relu(self.fusion_layer(x_fused))
        
        # 4. Contextualize with Event Features
        x_event = F.relu(self.event_encoder(event_features))
        x_event_expanded = x_event.unsqueeze(0).expand(N, -1)
        
        # 5. Score Nodes
        combined = torch.cat([x_nodes, x_event_expanded], dim=1)
        logits = self.scorer(combined).squeeze(-1)
        
        # 6. Autoregressive Masking
        logits = logits.masked_fill(seed_mask.bool(), float('-1e9'))
        
        return F.log_softmax(logits, dim=0)