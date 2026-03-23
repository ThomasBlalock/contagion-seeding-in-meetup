# process_graphs.py
"""
This file contains a script used to process the bipartite graph into a multi-scale bakboned
simple weighted coocurrance graph and a seperate 2-simplex weighted coocurrance graph.
"""

import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import itertools
from collections import defaultdict

def process_multiplex_graph(bipartite_graph, user_nodes, alpha=0.05, max_event_size=50):
    """
    Ingests a NetworkX bipartite graph and outputs PyG-ready 1-simplex and 
    2-simplex tensors, utilizing a multiscale backbone for noise reduction.
    
    Args:
        bipartite_graph: nx.Graph containing both user and event nodes.
        user_nodes: List of user node IDs (must match Bipartite Graph exactly).
        alpha: Statistical significance threshold for the Disparity Filter.
        max_event_size: Hard cutoff for 2-simplex combinatorics.
        
    Returns:
        edge_index_1, edge_attr_1, edge_index_2, edge_attr_2
    """
    
    # --- STEP 1: Node Index Alignment ---
    print("Mapping user nodes to contiguous indices...")
    user_to_idx = {u: i for i, u in enumerate(user_nodes)}
    
    event_nodes = set(bipartite_graph.nodes()) - set(user_nodes)
    event_to_users_dict = {}
    
    for event in event_nodes:
        # Convert raw NetworkX IDs into contiguous PyG integer indices
        attendees = [user_to_idx[u] for u in bipartite_graph.neighbors(event)]
        event_to_users_dict[event] = attendees

    # --- STEP 2: Sparse Projection & Disparity Filter (1-Simplices) ---
    print("Building sparse biadjacency matrix...")
    B = nx.bipartite.biadjacency_matrix(bipartite_graph, row_order=user_nodes, weight=None)

    print("Projecting co-occurrence matrix...")
    C = B.dot(B.T)
    C.setdiag(0)
    C.eliminate_zeros()

    print("Applying vectorized Disparity Filter...")
    C_csr = C.tocsr()
    C_coo = C.tocoo()

    rows = C_coo.row
    cols = C_coo.col
    weights = C_coo.data

    degrees = np.diff(C_csr.indptr)
    strengths = np.array(C_csr.sum(axis=1)).flatten()

    p_ij = weights / strengths[rows]
    k_minus_1_i = np.maximum(degrees[rows] - 1, 1) 
    alpha_ij = (1 - p_ij) ** k_minus_1_i

    p_ji = weights / strengths[cols]
    k_minus_1_j = np.maximum(degrees[cols] - 1, 1)
    alpha_ji = (1 - p_ji) ** k_minus_1_j

    mask = (alpha_ij < alpha) | (alpha_ji < alpha)

    filtered_rows = rows[mask]
    filtered_cols = cols[mask]
    filtered_weights = weights[mask]
    
    print(f"Backbone Filter: Reduced 1-simplices from {len(weights)//2} to {len(filtered_weights)//2}")

    edge_index_1 = torch.tensor(np.vstack((filtered_rows, filtered_cols)), dtype=torch.long)
    edge_attr_1 = torch.tensor(filtered_weights, dtype=torch.float).view(-1, 1)

    # --- STEP 3: Backbone-Conditioned Simplicial Closure (2-Simplices) ---
    print("Building lookup set for significant 1-simplices...")
    # Fast O(1) lookup for triplet validation
    significant_1_simplices = {
        tuple(sorted((u.item(), v.item()))) 
        for u, v in zip(edge_index_1[0], edge_index_1[1])
    }
    
    edges_2_dict = defaultdict(int)
    total_triplets = 0
    
    print("Extracting and filtering 2-simplices...")
    for event_id, users in event_to_users_dict.items():
        n = len(users)
        if not (3 <= n <= max_event_size):
            continue
            
        for u, v, w in itertools.combinations(sorted(users), 3):
            total_triplets += 1
            if (
                ((u, v) in significant_1_simplices) and 
                ((v, w) in significant_1_simplices) and 
                ((u, w) in significant_1_simplices)
            ):
                edges_2_dict[(u, v, w)] += 1
                
    print(f"2-Simplex Filter: Evaluated {total_triplets} triplets, retained {len(edges_2_dict)}")

    # --- STEP 4: Format 2-Simplices for PyTorch Geometric ---
    edge_index_2 = []
    edge_attr_2 = []
    
    for (u, v, w), weight in edges_2_dict.items():
        clique_edges = [[u, v], [v, u], [u, w], [w, u], [v, w], [w, v]]
        edge_index_2.extend(clique_edges)
        edge_attr_2.extend([[weight]] * 6)
        
    if not edge_index_2:
        edge_index_2 = torch.empty((2, 0), dtype=torch.long)
        edge_attr_2 = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long).t().contiguous()
        edge_attr_2 = torch.tensor(edge_attr_2, dtype=torch.float)
        
    return edge_index_1, edge_attr_1, edge_index_2, edge_attr_2