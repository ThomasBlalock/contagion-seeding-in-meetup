# TODO: boilerplate code - fix as needed
import torch
from torch.utils.data import Dataset
import os

class MultiplexImitationDataset(Dataset):
    def __init__(self, data_dir, use_dummy_embeddings=True, num_nodes=2000, node_dim=384, event_dim=384):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.use_dummy_embeddings = use_dummy_embeddings
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.event_dim = event_dim

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data_dict = torch.load(file_path)
        
        if self.use_dummy_embeddings:
            node_features = torch.randn(self.num_nodes, self.node_dim)
            event_features = torch.randn(self.event_dim)
        else:
            # Implement real loading logic here
            pass 
            
        state = {
            "node_features": node_features,
            "edge_index_1": data_dict["edge_index_1"],
            "edge_attr_1": data_dict["edge_attr_1"],
            "edge_index_2": data_dict["edge_index_2"],
            "edge_attr_2": data_dict["edge_attr_2"],
            "event_features": event_features,
            "seed_mask": data_dict["seed_mask"]
        }
        
        return state, data_dict["target_distribution"]