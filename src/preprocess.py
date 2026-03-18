# preprocess.py
# TODO: This is all boilerplate code. Review and change to work as needed

# IN: edges, hyperedges, node embeddings

# OUT: dataloader, SCM simulator

import torch
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
from copy import deepcopy

class ImitationDataGenerator:
    def __init__(self, simulator_template, num_mc_trials=50):
        self.sim_template = simulator_template
        self.num_mc_trials = num_mc_trials
        self.N = simulator_template.N

    def _evaluate_candidate(self, candidate_node):
        """Runs Monte Carlo trials for a specific candidate addition."""
        total_cascade = 0
        for _ in range(self.num_mc_trials):
            # Create a fresh simulator clone for this trial
            sim = deepcopy(self.sim_template)
            sim.current_state[candidate_node] = 1 # Add candidate to seed set
            
            rho_history = sim.run(t_max=50)
            # Cascade size is the final density * N
            total_cascade += rho_history[-1] * self.N 
            
        return candidate_node, total_cascade / self.num_mc_trials

    def _extract_2_simplices(self):
        """
        Converts the triangles list into a flattened edge_index of pairwise cliques.
        If i is in a triangle with (j, k), edges (i,j), (j,k), (i,k) are formed.
        """
        edges_2 = set()
        for i, neighbors in enumerate(self.sim_template.triangles):
            for j, k in neighbors:
                # Add all 3 undirected edges of the triangle
                clique = [(i, j), (j, i), (i, k), (k, i), (j, k), (k, j)]
                edges_2.update(clique)
        
        if not edges_2:
            return torch.empty((2, 0), dtype=torch.long)
            
        edge_index_2 = torch.tensor(list(edges_2), dtype=torch.long).t().contiguous()
        return edge_index_2

    def generate_step_target(self, current_seed_set, output_path, workers=8):
        self.sim_template.current_state = np.zeros(self.N, dtype=int)
        self.sim_template.current_state[current_seed_set] = 1
        
        valid_candidates = [v for v in range(self.N) if v not in current_seed_set]
        expected_cascades = torch.zeros(self.N)

        with Pool(workers) as p:
            results = p.map(self._evaluate_candidate, valid_candidates)
            
        for node, avg_cascade in results:
            expected_cascades[node] = avg_cascade
            
        expected_cascades[current_seed_set] = float('-inf')
        target_distribution = F.softmax(expected_cascades, dim=0)

        # Build edge_index_1 (1-simplices)
        edges_1 = [(i, j) for i, neighbors in enumerate(self.sim_template.links) for j in neighbors]
        edge_index_1 = torch.tensor(edges_1, dtype=torch.long).t().contiguous()
        
        # Build edge_index_2 (2-simplices)
        edge_index_2 = self._extract_2_simplices()

        # Mocking edge attributes (replace with actual logic based on your dataset)
        edge_attr_1 = torch.ones((edge_index_1.size(1), 1)) 
        edge_attr_2 = torch.ones((edge_index_2.size(1), 1))

        seed_mask = torch.zeros(self.N)
        seed_mask[current_seed_set] = 1.0

        data_dict = {
            "edge_index_1": edge_index_1,
            "edge_attr_1": edge_attr_1,
            "edge_index_2": edge_index_2,
            "edge_attr_2": edge_attr_2,
            "seed_mask": seed_mask,
            "target_distribution": target_distribution,
        }
        
        torch.save(data_dict, output_path)