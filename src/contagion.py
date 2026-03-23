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


class RSCGenerator:
    def __init__(self, k_avg, k_delta_avg, N=2000):
        self.N = N
        self.k_avg = k_avg
        self.k_delta_avg = k_delta_avg

        self.p_1 = (k_avg - 2*k_delta_avg) / (N - 1 - 2*k_delta_avg) 
        self.p_delta = (2*k_delta_avg) / ((N - 1)*(N - 2))

        self.links = [[] for _ in range(N)] # [[3,1,0],[],[0,6,92],...]
        self.triangles = [[] for _ in range(N)] # [[(3,1),)(7,32)],[],[(6,92)],...]

    def generate(self, seed=None):
        """Generates the topology using an optimized sampling approach"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._generate_1_simplices()
        self._generate_2_simplices()
        return self.links, self.triangles
        
    def _generate_1_simplices(self):
        """edge sampling from binomial"""
        print(f"Sampling edges with p_1 = {self.p_1:.8f}")
        num_edges = np.random.binomial(math.comb(self.N, 2), self.p_1)
        added_edges = set()
        
        # Rejection sampling
        node_list = range(self.N)
        while len(added_edges) < num_edges:
            i, j = random.sample(node_list, 2)
            edge = (i, j) if i < j else (j, i)
            
            if edge not in added_edges:
                added_edges.add(edge)
                self.links[i].append(j)
                self.links[j].append(i)
            
            print(f"\rEdges sampled: {len(added_edges)}/{num_edges}", end="")
        print()

    def _generate_2_simplices(self):
        """triangle hyperedge sampling from binomial"""
        print(f"Sampling triangles with p_delta = {self.p_delta:.8f}")
        num_triangles = np.random.binomial(math.comb(self.N, 3), self.p_delta)
        added_triangles = set()
        
        # Rejection sampling
        node_list = range(self.N)
        while len(added_triangles) < num_triangles:
            nodes = tuple(sorted(random.sample(node_list, 3)))
            if nodes not in added_triangles:
                added_triangles.add(nodes)
                i, j, k = nodes
                self.triangles[i].append((j, k))
                self.triangles[j].append((i, k))
                self.triangles[k].append((i, j))
            
            print(f"\rTriangles sampled: {len(added_triangles)}/{num_triangles}", end="")
        print()


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
        
        Args:
            edge_index_1: Tensor of shape (2, E1) for pairwise edges.
            edge_index_2: Tensor of shape (2, E2) for flattened 2-simplices.
            user_nodes: List of original user IDs. Index in list == simulator node_id.
        """
        self.user_nodes = user_nodes
        self.N = len(user_nodes)
        
        self.links = [[] for _ in range(self.N)]
        self.triangles = [[] for _ in range(self.N)]

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
        edge_set = set(zip(src, dst))
        
        # Build an intermediate adjacency list for the 2-simplex edges
        adj_2 = [[] for _ in range(self.N)]
        for u, v in zip(src, dst):
            adj_2[u].append(v)
            
        # Reconstruct true (j, k) triangles for node i
        for i in range(self.N):
            neighbors = adj_2[i]
            # Check all unique pairs of neighbors
            for idx_j in range(len(neighbors)):
                for idx_k in range(idx_j + 1, len(neighbors)):
                    j = neighbors[idx_j]
                    k = neighbors[idx_k]
                    # If the neighbors are connected in the 2-simplex graph, it forms a triangle
                    if (j, k) in edge_set:
                        self.triangles[i].append((j, k))

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


# %%
# Example Usage

# Initialize the RSCGenerator
rsc_generator = RSCGenerator(k_avg=20, k_delta_avg=6, N=2000)
# Generate the topology
links, triangles = rsc_generator.generate(seed=42)

# Initialize the SCMSimulator
initial_infected = RandomSeeding(N=2000, rho_0=0.05).seed()
simulator = SCMSimulator(links, triangles, initial_infected, beta=0.03, beta_delta=0.1, mu=0.01)

# Run the simulation and get density (rho) history (density/rho = fraction of infected nodes)
rho_history = simulator.run(t_max=200)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(range(len(rho_history)), rho_history)
plt.xlabel('Time')
plt.ylabel('Rho')
plt.show()