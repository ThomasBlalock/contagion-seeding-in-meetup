# preprocess.py
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from collections import deque
import random


class VectorizedSCMSimulator:
    def __init__(self, num_nodes, edge_index_1, triangles_list, beta_tensor, beta_delta_tensor, num_mc_trials=50, device='cuda'):
        """
        Fully vectorized SCM Simulator running M trials in parallel.
        """
        self.N = num_nodes
        self.M = num_mc_trials
        self.device = device
        
        self.beta = beta_tensor.to(device)
        self.beta_delta = beta_delta_tensor.to(device)
        
        # 1-Simplex Sparse Adjacency Matrix
        vals_1 = torch.ones(edge_index_1.shape[1], device=device)
        self.A_1 = torch.sparse_coo_tensor(edge_index_1.to(device), vals_1, (self.N, self.N)).coalesce()
        
        # 2-Simplex Tensor Setup for Vectorized Gathering
        if triangles_list:
            tri_tensor = torch.tensor(triangles_list, dtype=torch.long, device=device).t() # [3, T]
            self.tri_i = tri_tensor[0]
            self.tri_j = tri_tensor[1]
            self.tri_k = tri_tensor[2]
        else:
            self.tri_i = self.tri_j = self.tri_k = torch.empty(0, dtype=torch.long, device=device)

    def simulate_until_target(self, initial_infected, infected_target, max_steps=50):
        """
        Runs M simulations until the infected_target is reached or max_steps is hit.
        Returns the average number of timesteps required.
        """
        state = torch.zeros((self.M, self.N), dtype=torch.bool, device=self.device)
        state[:, initial_infected] = True
        
        timesteps_to_target = torch.full((self.M,), max_steps, dtype=torch.float, device=self.device)
        target_reached = torch.zeros(self.M, dtype=torch.bool, device=self.device)
        
        for t in range(1, max_steps + 1):
            current_infected_counts = state.sum(dim=1)
            newly_reached = (current_infected_counts >= infected_target) & (~target_reached)
            
            if newly_reached.any():
                timesteps_to_target[newly_reached] = t
                target_reached = target_reached | newly_reached
                
            if target_reached.all():
                break

            # 1-Simplex Exposures (m): [M, N]
            float_state = state.float()
            m = torch.sparse.mm(self.A_1, float_state.t()).t() 
            
            # 2-Simplex Exposures (n): [M, N]
            n = torch.zeros((self.M, self.N), dtype=torch.float, device=self.device)
            if self.tri_i.numel() > 0:
                # OOM FIX: Iterate over M trials to avoid [M, T] dense tensor materialization
                for m_idx in range(self.M):
                    infected_j = state[m_idx, self.tri_j] # [T]
                    infected_k = state[m_idx, self.tri_k] # [T]
                    infected_jk = (infected_j & infected_k).float()
                    
                    # Scatter add into the specific trial's node array
                    n[m_idx].scatter_add_(0, self.tri_i, infected_jk)

            # Compute infection probabilities: [M, N]
            prob_survival_1 = torch.pow(1.0 - self.beta.unsqueeze(0), m)
            prob_survival_2 = torch.pow(1.0 - self.beta_delta.unsqueeze(0), n)
            p_inf = 1.0 - (prob_survival_1 * prob_survival_2)
            
            p_inf[state] = 0.0
            
            rand_rolls = torch.rand((self.M, self.N), device=self.device)
            new_infections = rand_rolls < p_inf
            state = state | new_infections
            
            if not new_infections.any():
                break

        return timesteps_to_target.mean().item()

class ImitationDataGenerator:
    def __init__(self, num_nodes=None, edge_index_1=None, triangles_list=None, susceptibility_func=None, seeding_func=None,
                 num_mc_trials=50, top_n=30, infected_target=10, max_sim_steps=50, device='cpu'):
        
        if num_nodes is None or edge_index_1 is None or triangles_list is None or susceptibility_func is None or seeding_func is None:
            print("Num nodes:", num_nodes)
            print("Edge index 1:", edge_index_1)
            print("Triangles list:", triangles_list)
            print("Susceptibility func:", susceptibility_func)
            print("Seeding func:", seeding_func)
            raise ValueError("All parameters must be provided to initialize ImitationDataGenerator.")
        
        self.num_nodes = num_nodes
        self.seeding_func = seeding_func
        self.top_n = top_n
        self.infected_target = infected_target
        self.max_sim_steps = max_sim_steps
        
        # Precompute susceptibilities for the vectorized simulator
        print("Precomputing node susceptibilities...")
        beta_list, beta_delta_list = [], []
        for i in range(num_nodes):
            b, bd = susceptibility_func(i)
            beta_list.append(b)
            beta_delta_list.append(bd)
            
        self.beta_tensor = torch.tensor(beta_list, dtype=torch.float)
        self.beta_delta_tensor = torch.tensor(beta_delta_list, dtype=torch.float)
        
        self.simulator = VectorizedSCMSimulator(
            num_nodes=num_nodes,
            edge_index_1=edge_index_1,
            triangles_list=triangles_list,
            beta_tensor=self.beta_tensor,
            beta_delta_tensor=self.beta_delta_tensor,
            num_mc_trials=num_mc_trials,
            device=device
        )

    def generate(self, event_id=None, num_iter=100, max_seeds_per_iter=5, expand_best_n=3, expand_random_n=3, sampling_randomness=0.5):
        """
        Executes the main MCMC rollout loop with bounded tree expansion.
        """
        if event_id is None:
            raise ValueError("event_id must be provided to generate imitation data for a specific event.")

        dataset = []
        
        for iteration in tqdm(range(num_iter), desc="Generating Imitation Data", postfix={"MC Simulations": self.simulator.M}):
            initial_seed = np.random.randint(0, self.num_nodes)
            queue = deque([([initial_seed], 0)])
            visited_states = set()
            
            while queue:
                current_seeds, step = queue.popleft()
                
                state_hash = frozenset(current_seeds)
                if state_hash in visited_states:
                    continue
                visited_states.add(state_hash)
                
                if step >= max_seeds_per_iter:
                    continue
                    
                candidates_with_scores = self.seeding_func(current_seeds)
                top_candidates = [c[0] for c in candidates_with_scores[:int(self.top_n * sampling_randomness)]]
                top_candidates += random.sample([c[0] for c in candidates_with_scores[int(self.top_n * sampling_randomness):]],
                                               k=int(self.top_n * (1-sampling_randomness)))

                candidate_metrics = []
                for candidate in top_candidates:
                    if candidate in current_seeds:
                        continue
                        
                    sim_seeds = current_seeds + [candidate]
                    avg_timesteps = self.simulator.simulate_until_target(
                        initial_infected=sim_seeds,
                        infected_target=self.infected_target,
                        max_steps=self.max_sim_steps
                    )
                    candidate_metrics.append((candidate, avg_timesteps))
                
                if not candidate_metrics:
                    continue
                    
                raw_targets = {
                    cand: steps
                    for cand, steps in candidate_metrics
                }
                
                # Attach event context to the dataset
                dataset.append({
                    'event_id': event_id, 
                    'current_seeds': list(current_seeds), 
                    'candidate_targets': raw_targets
                })
                
                candidate_metrics.sort(key=lambda x: x[1])
                best_timesteps = candidate_metrics[0][1]
                if best_timesteps <= 1.0:
                    continue 
                
                best_to_expand = [c[0] for c in candidate_metrics[:expand_best_n]]
                remaining_candidates = [c[0] for c in candidate_metrics[expand_best_n:]]
                random_to_expand = []
                
                if expand_random_n > 0 and remaining_candidates:
                    num_to_sample = min(expand_random_n, len(remaining_candidates))
                    random_to_expand = random.sample(remaining_candidates, num_to_sample)
                
                nodes_to_expand = best_to_expand + random_to_expand
                for node in nodes_to_expand:
                    queue.append((current_seeds + [node], step + 1))
                    
        return dataset

import torch
from torch.utils.data import Dataset, DataLoader
import os

import torch
from torch.utils.data import Dataset

class StaticGraphSeedingDataset(Dataset):
    def __init__(self, imitation_dataset, num_nodes, event_features_tensor, max_sim_steps=50, scaling_mode='inverse', decay_rate=0.1):
        self.data_list = imitation_dataset
        self.num_nodes = num_nodes
        self.event_features_tensor = event_features_tensor
        self.max_sim_steps = max_sim_steps
        self.scaling_mode = scaling_mode
        self.decay_rate = decay_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        x = torch.zeros((self.num_nodes, 1), dtype=torch.float)
        if item['current_seeds']:
            x[item['current_seeds']] = 1.0
            
        y = torch.zeros(self.num_nodes, dtype=torch.float)
        candidate_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        for cand_node, raw_steps in item['candidate_targets'].items():
            # If the simulation hit max_steps, it failed to reach the target. 
            # Force the score to 0 to heavily penalize dead-end seeds.
            if raw_steps >= self.max_sim_steps:
                scaled_score = 0.0
            else:
                if self.scaling_mode == 'inverse':
                    # Max score 1.0 at t=1, asymptotically approaches 0
                    scaled_score = 1.0 / max(1.0, float(raw_steps))
                
                elif self.scaling_mode == 'exponential':
                    # Max score 1.0 at t=1, decays based on decay_rate
                    scaled_score = np.exp(-self.decay_rate * (max(1.0, float(raw_steps)) - 1.0))
                
                elif self.scaling_mode == 'linear':
                    scaled_score = max(0.0, 1.0 - (raw_steps / self.max_sim_steps))
                    
                else:
                    raise ValueError(f"Unknown scaling mode: {self.scaling_mode}")

            y[cand_node] = scaled_score
            candidate_mask[cand_node] = True
            
        # Fetch dense event features using the integer ID
        event_idx = item['event_id']
        event_feat = self.event_features_tensor[event_idx]
            
        return x, y, candidate_mask, event_feat

def collate_static_graph_signals(batch):
    x_list, y_list, mask_list, event_feat_list = zip(*batch)
    return torch.stack(x_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(event_feat_list)

import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

def build_production_dataloader(imitation_dataset, user_idx, event_idx, data_dir="data", batch_size=32, shuffle=True, max_sim_steps=50):
    num_nodes = len(user_idx)
    
    print("Loading static graph topology...")
    static_graph = {
        'edge_index_1': torch.load(os.path.join(data_dir, "edge_index_simple.pt")),
        'edge_attr_1': torch.load(os.path.join(data_dir, "edge_attr_simple.pt")),
        'edge_index_2': torch.load(os.path.join(data_dir, "edge_index_hyper.pt")),
        'edge_attr_2': torch.load(os.path.join(data_dir, "edge_attr_hyper.pt"))
    }
    
    print("Aligning static user features...")
    df_features = pd.read_csv(os.path.join(data_dir, "user_features.csv"), index_col=0)
    
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    ordered_member_ids = [user_idx[i] for i in range(num_nodes)]
    aligned_features_df = df_features.reindex(ordered_member_ids).fillna(0.0)
    
    # FIX: Explicitly cast the underlying numpy array to a homogenous float32 block
    static_graph['x_static'] = torch.tensor(aligned_features_df.values.astype(np.float32), dtype=torch.float)
    
    print("Aligning event features...")
    df_event_features = pd.read_csv(os.path.join(data_dir, "event_features.csv"), index_col=0)
    
    df_event_features = df_event_features.apply(pd.to_numeric, errors='coerce')
    aligned_event_features = df_event_features.reindex(event_idx).fillna(0.0)
    
    # FIX: Explicitly cast the underlying numpy array to a homogenous float32 block
    event_features_tensor = torch.tensor(aligned_event_features.values.astype(np.float32), dtype=torch.float)
    
    dataset = StaticGraphSeedingDataset(imitation_dataset, num_nodes, event_features_tensor, max_sim_steps=max_sim_steps)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_static_graph_signals,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, static_graph