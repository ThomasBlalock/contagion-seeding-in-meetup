# %%
import numpy as np
import random
import math


class NodeSusceptibilityModel:
    def __init__(self, node_features, base_beta=0.03, base_beta_delta=0.1):
        """
        Stores the feature matrix and parameters needed to calculate probabilities.
        """
        self.node_features = node_features
        self.base_beta = base_beta
        self.base_beta_delta = base_beta_delta

    def __call__(self, node_id):
        """
        Calculates beta and beta_delta for a specific node_id.
        """
        # Example logic: scale baseline probabilities by a specific node feature.
        # Replace this with your actual Phase 2.a Network-Unaware model forward pass.
        feature_multiplier = self.node_features[node_id][0].item() 
        
        beta = self.base_beta * feature_multiplier
        beta_delta = self.base_beta_delta * feature_multiplier
        
        # Clamp to valid probability bounds
        beta = max(0.0, min(1.0, beta))
        beta_delta = max(0.0, min(1.0, beta_delta))
        
        return beta, beta_delta

class RandomSeeding:
    def __init__(self, N, rho_0):
        self.N = N
        self.rho_0 = rho_0

    def seed(self):
        num_initial_infected = int(self.N * self.rho_0)
        initial_infected = random.sample(range(self.N), num_initial_infected)
        return initial_infected

import numpy as np
import random
import math
import torch

class MultiplexTopologyAdapter:
    def __init__(self, edge_index_1, edge_index_2, user_nodes):
        """
        Parses PyTorch Geometric tensors into the adjacency lists required by the simulator.
        """
        self.user_nodes = user_nodes
        self.N = len(user_nodes)
        
        self.links = [[] for _ in range(self.N)]
        self.triangles = [] # Flattened for VectorizedSCMSimulator compatibility

        self._build_links(edge_index_1)
        self._build_triangles(edge_index_2)

    def _build_links(self, edge_index_1):
        if edge_index_1.numel() == 0:
            return
            
        src, dst = edge_index_1.numpy()
        for u, v in zip(src, dst):
            self.links[u].append(v)

    def _build_triangles(self, edge_index_2):
        if edge_index_2.numel() == 0:
            return
            
        src, dst = edge_index_2.numpy()
        
        # 1. Build fast lookup sets
        adj_sets = [set() for _ in range(self.N)]
        for u, v in zip(src, dst):
            adj_sets[u].add(v)
            
        flat_triangles = []
        
        # 2. Use set intersection to find cliques (O(E * min(d_u, d_v)))
        for i in range(self.N):
            neighbors = adj_sets[i]
            for j in neighbors:
                if j <= i: # Symmetry breaking: only evaluate each edge once
                    continue
                    
                # The intersection of neighbors yields the 3rd node in the triangle
                common_neighbors = neighbors.intersection(adj_sets[j])
                
                for k in common_neighbors:
                    if k <= j: # Symmetry breaking: ensure i < j < k
                        continue
                        
                    # Found a unique triangle (i, j, k). 
                    # The vectorized simulator needs a target exposure mapping for all 3 nodes.
                    flat_triangles.append((i, j, k))
                    flat_triangles.append((j, i, k))
                    flat_triangles.append((k, i, j))
                    
        self.triangles = flat_triangles

    def get_original_id(self, node_id):
        return self.user_nodes[node_id]


class SCMSimulator:
    def __init__(self, links, triangles, initial_infected, susceptibility_func, mu=0.0):
        """
        Args:
            links: 1-simplex adjacency list.
            triangles: 2-simplex adjacency list.
            initial_infected: List of integer node IDs to start infected.
            susceptibility_func: Callable that takes (node_id) and returns (beta, beta_delta).
            mu: Recovery probability (default 0.0 for monotonic SI contagion).
        """
        self.links = links
        self.triangles = triangles
        self.N = len(links)

        self.susceptibility_func = susceptibility_func
        self.mu = mu
        
        self.current_state = np.zeros(self.N, dtype=int)
        self.current_state[initial_infected] = 1
        
        self.rho_history = [np.mean(self.current_state)]
    
    def run(self, t_max):
        for t in range(t_max):
            self._step()
            if self.stable_state(): 
                self.rho_history.extend([self.rho_history[-1]] * (t_max - t - 1))
                break
                
        return self.rho_history

    def stable_state(self):
        if self.rho_history[-1] == 0.0:
            return True
        if len(self.rho_history) > 100 and np.isclose(self.rho_history[-1], self.rho_history[-100], atol=1e-5):
            return True
        return False
        
    def _step(self):
        next_state = np.copy(self.current_state)
        
        for i in range(self.N):
            if self.current_state[i] == 1: 
                if self.mu > 0 and random.random() < self.mu:
                    next_state[i] = 0
            else: 
                m = self._count_infected_links(i)
                n = self._count_infected_triangles(i)
                
                # Skip calculation if no social exposure
                if m == 0 and n == 0:
                    continue
                
                # Dynamic susceptibility fetch
                beta, beta_delta = self.susceptibility_func(i)
                
                p_inf = 1.0 - ((1.0 - beta)**m)*((1.0 - beta_delta)**n)
                
                if random.random() < p_inf:
                    next_state[i] = 1
                    
        self.current_state = next_state
        self.rho_history.append(np.mean(self.current_state))

    def _count_infected_links(self, i):
        return sum(self.current_state[neighbor] for neighbor in self.links[i])

    def _count_infected_triangles(self, i):
        count = 0
        for j, k in self.triangles[i]:
            if self.current_state[j] == 1 and self.current_state[k] == 1:
                count += 1
        return count
    

import numpy as np
from scipy.stats import wasserstein_distance

def calibrate_parameters(simulator_factory, historical_cascade_sizes, param_grid):
    """
    Finds beta_1 and beta_2 that minimize Wasserstein distance to ground truth.
    
    Args:
        simulator_factory: A lambda or function that returns a new SCMSimulator 
                           initialized with a specific beta pair.
        historical_cascade_sizes: List of final RSVP counts from your Meetup data.
        param_grid: List of (beta_1, beta_2) tuples to test.
    """
    best_dist = float('inf')
    best_params = None
    
    for b1, b2 in param_grid:
        sim_results = []
        
        # Run M trials to get a distribution for this parameter set
        for _ in range(50): 
            sim = simulator_factory(b1, b2)
            history = sim.run(t_max=100)
            # Final cascade size = final density * N
            sim_results.append(history[-1] * sim.N)
            
        dist = wasserstein_distance(historical_cascade_sizes, sim_results)
        
        if dist < best_dist:
            best_dist = dist
            best_params = (b1, b2)
            print(f"New Best! b1:{b1:.4f}, b2:{b2:.4f} | Dist: {dist:.4f}")
            
    return best_params