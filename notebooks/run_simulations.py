import pandas as pd
import numpy as np
import torch
import networkx as nx
import pickle
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
from contagion import MultiplexTopologyAdapter
from preprocess import ImitationDataGenerator
from seeder import SimplicialSeeder

# Configuration all in one place for easy adjustments
config = {
    "lam": 0.6, # base infection probability scaling factor for simple edges
    "lam_d": 1, # base infection probability scaling factor for hyperedges (triangles)
    "sample_num_events": 1, # Number of distinct events to generate data for (randomly sampled from the dataset)
    "init_params": {
        "num_mc_trials": 20, # Number of simulations per candidate seed to calculate avg timesteps
        "top_n": 30, # Number of seed nodes to consider at each iteration (from seeding function)
        "infected_target": 10, # Target number of infected nodes to reach in the simulation
        "max_sim_steps": 50, # Maximum steps to run the contagion simulation before considering it a failure (timeout)
        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    "generate_params": {
        "num_iter": 3, # Number of simulation iterations to perform for each event
        "max_seeds_per_iter": 2, # EXPONENTIAL FACTOR: KEEP LOW
        "expand_best_n": 3, # Number of best performing seed nodes to expand in the next iteration
        "expand_random_n": 3, # Number of randomly selected seed nodes to expand in the next iteration (for exploration)
        "sampling_randomness": 0.5 # Balance between testing promising candidates and exploring diverse (random) options
    }
}

edge_simple = torch.load("data/edge_index_simple.pt")
edge_hyper = torch.load("data/edge_index_hyper.pt")

config["init_params"]["edge_index_1"] = edge_simple

with open("data/user_idx.pkl", "rb") as f:
    user_idx = pickle.load(f)
with open("data/event_idx.pkl", "rb") as f:
    idx_to_event = pickle.load(f)

class ContagionProbabilityLookup:
    """
    O(1) memory-backed lookup table for user-event contagion probabilities.
    """
    def __init__(self, probability_matrix, user_mapping, event_mapping):
        self.prob_matrix = probability_matrix
        self.user_to_idx = user_mapping
        self.event_to_idx = event_mapping

    def __call__(self, user_id, event_id) -> float:
        """
        Accepts the original user ID and event ID strings/ints and 
        returns the calibrated transmission probability.
        """
        if user_id not in self.user_to_idx or event_id not in self.event_to_idx:
            # Return baseline probability or 0.0 if the node isn't found
            return 0.0 
        
        u_idx = self.user_to_idx[user_id]
        e_idx = self.event_to_idx[event_id]
        return self.prob_matrix[u_idx, e_idx].item()
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
config["init_params"]["num_nodes"] = adpt.N # Update num_nodes based on adapter
config["init_params"]["triangles_list"] = adpt.triangles # Update triangles list based on adapter
seeder = SimplicialSeeder(adpt.N, adpt.links, adpt.triangles, top_n=config["init_params"]["top_n"])
config["init_params"]["seeding_func"] = seeder
print("Links and triangles parsed:")
print(f"Node 0 links: {adpt.links[0]}")
print(f"Node 0 triangles: {adpt.triangles[0]}")
print("Total nodes:", adpt.N)
event_ids = np.random.choice(
    idx_to_event, 
    size=min(config["sample_num_events"], len(idx_to_event)), replace=False)
combined_data = None
for event_id in event_ids:
    lam = config["lam"]
    lam_d = config["lam_d"]
    def sus_func(node_id):
        original_user_id = user_idx[node_id]
        original_event_id = idx_to_event[event_id]
        p = prob_lookup(user_id=original_user_id, event_id=original_event_id)
        return lam*(p), lam_d*(p)
    config["init_params"]["susceptibility_func"] = sus_func

    data_gen = ImitationDataGenerator(**config["init_params"])
    config["generate_params"]["event_id"] = event_id
    data = data_gen.generate(**config["generate_params"])
    combined_data = combine_imitation_data(combined_data, data)

import pickle

with open("data/imitation_data.pkl", "wb") as f:
    pickle.dump(combined_data, f)
        
print("Complete. Plotting timestep distribution...")

import matplotlib.pyplot as plt
import numpy as np

def plot_timestep_distribution(combined_data, max_steps=50):
    """
    Extracts raw timesteps from the MCMC rollout data and plots the distribution.
    Assumes 'candidate_targets' contains raw timesteps.
    """
    all_steps = []
    
    for rollout in combined_data:
        for node, steps in rollout['candidate_targets'].items():
            all_steps.append(steps)

    if not all_steps:
        print("No timestep data found.")
        return

    all_steps = np.array(all_steps)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Align bins so integers fall in the center of the bars
    bins = np.arange(1, max_steps + 2) - 0.5 
    
    ax.hist(all_steps, bins=bins, edgecolor='black', alpha=0.75, color='steelblue')
    
    mean_steps = np.mean(all_steps)
    median_steps = np.median(all_steps)
    failed_cascades = np.sum(all_steps >= max_steps)
    
    ax.axvline(mean_steps, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_steps:.2f}')
    ax.axvline(median_steps, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_steps:.2f}')
    
    ax.set_title('Distribution of Timesteps Required to Reach Infection Target')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Frequency')
    ax.set_xticks(range(1, max_steps + 1, max(1, max_steps // 10)))
    
    # Flag timeouts indicating seeds that failed to reach the target threshold
    if failed_cascades > 0:
        ax.text(0.95, 0.5, f'Timeouts (≥{max_steps}): {failed_cascades:,}', 
                transform=ax.transAxes, ha='right', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Execution
plot_timestep_distribution(combined_data, max_steps=50)