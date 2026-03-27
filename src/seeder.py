import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

class SimplicialSeeder:
    """
    Seeding strategies for Simplicial Contagion Models.
    Designed to be stateless with respect to tree expansion to support BFS/MCMC rollouts safely.
    """
    def __init__(self, num_nodes: int, links: List[List[int]], flat_triangles: List[Tuple[int, int, int]]):
        self.N = num_nodes
        self.links = links
        
        # 1. Reconstruct the nested adjacency list expected by the CELF proxy
        self.triangles = [[] for _ in range(self.N)]
        for i, j, k in flat_triangles:
            self.triangles[i].append((j, k))

        # 2. Convert flat triangles into a unique [3, T] tensor for vectorization
        triplets = set()
        for i, j, k in flat_triangles:
            triplets.add(tuple(sorted([i, j, k])))
        
        if triplets:
            self.triangles_tensor = torch.tensor(list(triplets), dtype=torch.long).t()
        else:
            self.triangles_tensor = torch.empty((3, 0), dtype=torch.long)

        self._precompute_simplicial_degree()
        self._precompute_co_seeding_edges()

        # Pass the reconstructed nested list to CELF
        self.celf_optimizer = OptimizedSimplicialSeeding(self.N, 0.0, self.links, self.triangles)

    def __call__(self, current_seeds: List[int], beta: float = 0.1, beta_delta: float = 0.2) -> List[Tuple[int, float]]:
        """
        Executes all methods and unifies outputs into the `[(node, score), ...]` 
        format expected by ImitationDataGenerator.
        """
        candidates = []
        seen = set(current_seeds)
        
        # Interleave candidates from different strategies to ensure structural diversity.
        # Assigned decreasing mock scores to maintain the generated rank order.
        sd_nodes = self.simplicial_degree_centrality(current_seeds, top_k=5)
        tc_nodes = self.triangle_co_seeding(current_seeds, top_k=5)
        celf_nodes = self.celf_proxy_seeding(current_seeds, top_k=5, beta=beta, beta_delta=beta_delta)
        rs_nodes = self.random_sampling(current_seeds, top_k=5)

        max_len = max(len(sd_nodes), len(tc_nodes), len(celf_nodes), len(rs_nodes))
        current_score = 1.0
        score_decrement = 0.05

        for i in range(max_len):
            # Prioritize CELF and co-seeding
            for strat_list in (celf_nodes, tc_nodes, sd_nodes, rs_nodes): 
                if i < len(strat_list):
                    node = strat_list[i]
                    if node not in seen:
                        random_score = np.random.uniform(0.001, 0.999)
                        candidates.append((node, random_score))
                        seen.add(node)
                        current_score -= score_decrement
                        
        return candidates

    def _precompute_simplicial_degree(self):
        """O(N) vectorization for 2-simplex participation counts."""
        if self.triangles_tensor.numel() == 0:
            self.simplicial_ranking = []
            return
            
        self.node_tri_counts = torch.bincount(self.triangles_tensor.flatten(), minlength=self.N)
        self.simplicial_ranking = torch.argsort(self.node_tri_counts, descending=True).tolist()

    def _precompute_co_seeding_edges(self):
        """O(T \log T) vectorized edge overlap precomputation."""
        if self.triangles_tensor.numel() == 0:
            self.sorted_edges = []
            return
            
        e1 = self.triangles_tensor[[0, 1], :]
        e2 = self.triangles_tensor[[1, 2], :]
        e3 = self.triangles_tensor[[0, 2], :]
        
        all_edges = torch.cat([e1, e2, e3], dim=1)
        all_edges, _ = torch.sort(all_edges, dim=0) # Order invariant
        
        unique_edges, counts = torch.unique(all_edges, dim=1, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        
        # Stored as a list of [u, v] pairs ordered by triangle participation
        self.sorted_edges = unique_edges[:, sorted_indices].t().tolist()

    
    def simplicial_degree_centrality(self, current_seeds: List[int], top_k: int = 5) -> List[int]:
        """
        Rank nodes by their 2-simplex participation. 
        """
        current_set = set(current_seeds)
        candidates = []
        for node in self.simplicial_ranking:
            if node not in current_set:
                candidates.append(node)
                if len(candidates) == top_k:
                    break
        return candidates

    def triangle_co_seeding(self, current_seeds: List[int], top_k: int = 5) -> List[int]:
        """
        Infers the required pairing state natively from the current_seeds list.
        This avoids state-mutation bugs during parallel tree expansion (MCMC).
        """
        current_set = set(current_seeds)
        candidates = []
        
        # Pass 1: Look for highly-ranked edges where EXACTLY ONE node is already seeded.
        # This completes the pair, instantly activating \beta_\Delta for shared neighbors.
        for u, v in self.sorted_edges:
            u_in, v_in = u in current_set, v in current_set
            
            if u_in and not v_in and v not in candidates:
                candidates.append(v)
            elif v_in and not u_in and u not in candidates:
                candidates.append(u)
                
            if len(candidates) >= top_k:
                return candidates

        # Pass 2: If no half-seeded pairs exist (or we need more candidates), 
        # seed the highest-ranked entirely unseeded edges.
        for u, v in self.sorted_edges:
            if u not in current_set and v not in current_set:
                if u not in candidates:
                    candidates.append(u)
                if len(candidates) >= top_k:
                    break
                if v not in candidates:
                    candidates.append(v)
                if len(candidates) >= top_k:
                    break

        return candidates

    def random_sampling(self, current_seeds: List[int], top_k: int = 5) -> List[int]:
        """Uniform random sampling of remaining nodes."""
        valid_nodes = list(set(range(self.N)) - set(current_seeds))
        if not valid_nodes:
            return []
        
        sample_size = min(top_k, len(valid_nodes))
        return np.random.choice(valid_nodes, size=sample_size, replace=False).tolist()
    
    def _build_adjacency_lists(self):
        """Converts PyTorch edge indices to Python lists for the CELF proxy."""
        self.links = [[] for _ in range(self.N)]
        for u, v in self.edge_index_1.t().tolist():
            self.links[u].append(v)
            
        self.triangles = [[] for _ in range(self.N)]
        for i, j, k in self.edge_index_2.t().tolist():
            self.triangles[i].append((j, k))
            self.triangles[j].append((i, k))
            self.triangles[k].append((i, j))

    def celf_proxy_seeding(self, current_seeds: List[int], top_k: int = 5, beta: float = 0.1, beta_delta: float = 0.2) -> List[int]:
        """Executes CELF proxy incrementally from the current MCMC/BFS state."""
        self.celf_optimizer.target_count = len(current_seeds) + top_k
        full_seed_set = self.celf_optimizer.seed_celf_proxy(beta, beta_delta, initial_seeds=current_seeds)
        
        # Isolate and return only the newly added nodes to maintain top_k structure
        return [node for node in full_seed_set if node not in current_seeds]
    

import numpy as np
import heapq

class OptimizedSimplicialSeeding:
    def __init__(self, N, rho_0, links, triangles):
        self.N = N
        self.rho_0 = rho_0
        self.links = links
        self.triangles = triangles
        self.target_count = int(self.N * self.rho_0)

    def _proxy_spread(self, seed_set, beta, beta_delta):
        """
        Deterministic 1-hop expected activation proxy.
        Calculates expected spread without executing multi-step Monte Carlo simulations.
        """
        if not seed_set:
            return 0.0

        spread = len(seed_set)
        
        m_counts = np.zeros(self.N, dtype=int)
        n_counts = np.zeros(self.N, dtype=int)
        
        for s in seed_set:
            # Count 1-simplex exposures
            for neighbor in self.links[s]:
                if neighbor not in seed_set:
                    m_counts[neighbor] += 1
                    
            # Count 2-simplex exposures
            for j, k in self.triangles[s]:
                if j in seed_set and k not in seed_set:
                    n_counts[k] += 1
                elif k in seed_set and j not in seed_set:
                    n_counts[j] += 1
                    
        # A triangle shared by two seeded nodes will be counted twice (once from each seeded node).
        n_counts = n_counts // 2 
        
        unseeded_mask = np.ones(self.N, dtype=bool)
        unseeded_mask[list(seed_set)] = False
        
        active_mask = unseeded_mask & ((m_counts > 0) | (n_counts > 0))
        
        if np.any(active_mask):
            p_activation = 1.0 - ((1.0 - beta)**m_counts[active_mask]) * ((1.0 - beta_delta)**n_counts[active_mask])
            spread += np.sum(p_activation)
            
        return spread

    def seed_celf_proxy(self, beta, beta_delta, initial_seeds=None):
        """
        Modified to calculate marginal gains on top of an existing seed state
        to support incremental rollout tracking.
        """
        seed_set = set(initial_seeds) if initial_seeds else set()
        
        if self.target_count <= len(seed_set):
            return list(seed_set)

        Q = [] 
        current_spread = self._proxy_spread(seed_set, beta, beta_delta) if seed_set else 0.0
        
        # 1. Initial pass - evaluate marginal gain relative to initial_seeds
        for node in range(self.N):
            if node in seed_set:
                continue
                
            seed_set.add(node)
            spread = self._proxy_spread(seed_set, beta, beta_delta)
            seed_set.remove(node)
            
            marginal_gain = spread - current_spread
            heapq.heappush(Q, (-marginal_gain, node, len(seed_set)))

        # 2. Lazy evaluation loop
        while len(seed_set) < self.target_count:
            neg_mg, node, last_update = heapq.heappop(Q)
            
            if last_update == len(seed_set):
                seed_set.add(node)
                current_spread += -neg_mg
            else:
                seed_set.add(node)
                new_spread = self._proxy_spread(seed_set, beta, beta_delta)
                seed_set.remove(node)
                
                marginal_gain = new_spread - current_spread
                heapq.heappush(Q, (-marginal_gain, node, len(seed_set)))

        return list(seed_set)