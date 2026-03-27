import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

class SimplicialSeeder:
    """
    Seeding strategies for Simplicial Contagion Models.
    Designed to be stateless with respect to tree expansion to support BFS/MCMC rollouts safely.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path.cwd().parent / "notebooks" / "data"
        else:
            data_dir = Path(data_dir)

        # Load topologies
        self.edge_index_1 = torch.load(data_dir / "edge_index_simple.pt")
        self.edge_attr_1 = torch.load(data_dir / "edge_attr_simple.pt")
        self.edge_index_2 = torch.load(data_dir / "edge_index_hyper.pt")
        self.edge_attr_2 = torch.load(data_dir / "edge_attr_hyper.pt")

        self.N = max(self.edge_index_1.max().item(), self.edge_index_2.max().item()) + 1

        self._precompute_simplicial_degree()
        self._precompute_co_seeding_edges()

    def __call__(self, current_seeds: List[int]) -> List[Tuple[int, float]]:
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
        rs_nodes = self.random_sampling(current_seeds, top_k=5)

        max_len = max(len(sd_nodes), len(tc_nodes), len(rs_nodes))
        current_score = 1.0
        score_decrement = 0.05

        for i in range(max_len):
            for strat_list in (tc_nodes, sd_nodes, rs_nodes): # Prioritize co-seeding pairs
                if i < len(strat_list):
                    node = strat_list[i]
                    if node not in seen:
                        candidates.append((node, current_score))
                        seen.add(node)
                        current_score -= score_decrement
                        
        return candidates

    def _precompute_simplicial_degree(self):
        """$O(N)$ vectorization for 2-simplex participation counts."""
        self.node_tri_counts = torch.bincount(self.edge_index_2.flatten(), minlength=self.N)
        self.simplicial_ranking = torch.argsort(self.node_tri_counts, descending=True).tolist()

    def _precompute_co_seeding_edges(self):
        """$O(T \log T)$ vectorized edge overlap precomputation."""
        e1 = self.edge_index_2[[0, 1], :]
        e2 = self.edge_index_2[[1, 2], :]
        e3 = self.edge_index_2[[0, 2], :]
        
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