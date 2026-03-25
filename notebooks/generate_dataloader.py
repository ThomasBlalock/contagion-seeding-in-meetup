import pandas as pd
import numpy as np
import torch
import networkx as nx
import pickle

edge_simple = torch.load("data/edge_index_simple.pt")
edge_hyper = torch.load("data/edge_index_hyper.pt")

with open("data/user_idx.pkl", "rb") as f:
    user_idx = pickle.load(f)

import sys
import os
sys.path.append('/home/blalo/uva-work/contagion-seeding-in-meetup/src')
from contagion import MultiplexTopologyAdapter
from preprocess import ImitationDataGenerator

print("Edge index for simple graph:", edge_simple.shape)
print("Edge index for hyper graph:", edge_hyper.shape)
adpt = MultiplexTopologyAdapter(edge_simple, edge_hyper, user_idx)
print("Links and triangles parsed:")
print(f"Node 0 links: {adpt.links[0]}")
print(f"Node 0 triangles: {adpt.triangles[0]}")
print("Total nodes:", adpt.N)
data_gen = ImitationDataGenerator(
    num_nodes=len(user_idx),
    edge_index_1=edge_simple,
    triangles_list=adpt.triangles,
    susceptibility_func=lambda x: (0.03, 0.1),
    seeding_func=lambda state: [(i, np.random.rand()) for i in range(100) if i not in state],
    num_mc_trials=50,
    top_n=30,
    infected_target=10,
    max_sim_steps=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

data = data_gen.generate(
    event_id=0,
    num_iter=1, 
    max_seeds_per_iter=2, # EXPONENTIAL FACTOR: KEEP LOW
    expand_best_n=2, 
    expand_random_n=2)
print(data)

import pickle

with open("data/imitation_data.pkl", "wb") as f:
    pickle.dump(data, f)

with open("data/event_idx.pkl", "rb") as f:
    event_idx = pickle.load(f)

from preprocess import build_production_dataloader

print("Building dataloader...")
dataloader, static_graph = build_production_dataloader(
    imitation_dataset=data, 
    user_idx=user_idx, 
    event_idx=event_idx,
    data_dir="data",
    batch_size=32,
    shuffle=True
)

print("Writing dataloader batch shapes to file...")
with open("dataloader_output_test.txt", "w") as f:
    f.write("Static Graph Topology & Features:\n")
    for k, v in static_graph.items():
        f.write(f"{k}: {v.shape}\n")
        
    f.write("\nFirst Batch Shapes:\n")
    for batch_idx, (seed_mask, y, eval_mask, event_feat) in enumerate(dataloader):
        f.write(f"Seed Mask: {seed_mask.shape}\n")
        f.write(f"Targets (y): {y.shape}\n")
        f.write(f"Eval Mask: {eval_mask.shape}\n")
        f.write(f"Event Features: {event_feat.shape}\n")
        break 
        
print("Complete. Check dataloader_output_test.txt")