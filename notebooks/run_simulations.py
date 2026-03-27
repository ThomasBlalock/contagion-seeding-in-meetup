import pandas as pd
import numpy as np
import torch
import networkx as nx
import pickle

edge_simple = torch.load("data/edge_index_simple.pt")
edge_hyper = torch.load("data/edge_index_hyper.pt")

with open("data/user_idx.pkl", "rb") as f:
    user_idx = pickle.load(f)
with open("data/event_idx.pkl", "rb") as f:
    idx_to_event = pickle.load(f)

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
from contagion import MultiplexTopologyAdapter
from preprocess import ImitationDataGenerator
from seeder import SimplicialSeeder

seeder = SimplicialSeeder()

with open("data/probability_lookup.pkl", "rb") as f:
    prob_lookup = pickle.load(f)
def sus_func():
    return prob_lookup()

def combine_imitation_data(existing_data, new_data):
    """
    Concatenates the MCMC rollout lists from multiple events.
    """
    if existing_data is None:
        return new_data
    return existing_data + new_data

print("Edge index for simple graph:", edge_simple.shape)
print("Edge index for hyper graph:", edge_hyper.shape)
adpt = MultiplexTopologyAdapter(edge_simple, edge_hyper, user_idx)
print("Links and triangles parsed:")
print(f"Node 0 links: {adpt.links[0]}")
print(f"Node 0 triangles: {adpt.triangles[0]}")
print("Total nodes:", adpt.N)
event_ids = [1, 4, 5]
combined_data = None
for event_id in event_ids:
    lam = 0.3
    lam_d = 0.95
    def sus_func(node_id):
        original_user_id = user_idx[node_id]
        original_event_id = idx_to_event[event_id]
        p = prob_lookup(user_id=original_user_id, event_id=original_event_id)
        return lam*(p), lam_d*(p)
    data_gen = ImitationDataGenerator(
        num_nodes=len(user_idx),
        edge_index_1=edge_simple,
        triangles_list=adpt.triangles,
        susceptibility_func=sus_func,
        seeding_func=seeder,
        num_mc_trials=50,
        top_n=30,
        infected_target=10,
        max_sim_steps=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    data = data_gen.generate(
        event_id=event_id,
        num_iter=1, 
        max_seeds_per_iter=2, # EXPONENTIAL FACTOR: KEEP LOW
        expand_best_n=2, 
        expand_random_n=2
        )
    combined_data = combine_imitation_data(combined_data, data)

import pickle

with open("data/imitation_data.pkl", "wb") as f:
    pickle.dump(combined_data, f)
        
print("Complete.")