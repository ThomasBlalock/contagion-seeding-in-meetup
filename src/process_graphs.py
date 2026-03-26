# %%
# process_graphs.py
"""
This file contains a script used to process the bipartite graph into a multi-scale bakboned
simple weighted coocurrance graph and a seperate 2-simplex weighted coocurrance graph.
"""

from anyio import Path
import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import itertools
from collections import defaultdict

def load_bipartite_artifacts(input_dir=""): # Took data out
    in_path = Path.cwd() # No input_dir for the moment SURRYYY AGAIN / input_dir
    G_bipartite = nx.read_graphml(in_path / "G_bipartite.graphml")
    
    with open(in_path / "member_nodes.csv", 'r') as f:
        member_nodes = [line.strip() for line in f]
        
    return G_bipartite, member_nodes


import torch
import numpy as np
import networkx as nx
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
        edge_index_1_torch, edge_attr_1_torch, edge_index_2_torch, edge_attr_2_torch, idx_to_user
    """
    
    # --- STEP 1: Node Index Alignment ---
    print("Mapping user nodes to contiguous indices...")
    user_to_idx = {u: i for i, u in enumerate(user_nodes)}
    
    # ADDED: Reverse mapping list (O(1) indexing, lower memory footprint)
    idx_to_user = list(user_nodes)

    event_nodes = [n for n, attr in bipartite_graph.nodes(data=True) if attr.get('type') == 'event']
    event_to_users_dict = {}

    print("Mapping user nodes to contiguous indices...")
    user_to_idx = {u: i for i, u in enumerate(user_nodes)}
    idx_to_user = list(user_nodes)
    
    # MODIFIED: Extract, filter, and sort deterministically to prevent indexing drift
    idx_to_event = sorted([n for n, attr in bipartite_graph.nodes(data=True) if attr.get('type') == 'event'])
    
    event_to_users_dict = {}
    for event in idx_to_event:
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

    edge_index_1_torch = torch.tensor(np.vstack((filtered_rows, filtered_cols)), dtype=torch.long)
    edge_attr_1_torch = torch.tensor(filtered_weights, dtype=torch.float).view(-1, 1)

    # --- STEP 3: Backbone-Conditioned Simplicial Closure (2-Simplices) ---
    print("Building lookup set for significant 1-simplices...")
    # Fast O(1) lookup for triplet validation
    significant_1_simplices = {
        tuple(sorted((u.item(), v.item()))) 
        for u, v in zip(edge_index_1_torch[0], edge_index_1_torch[1])
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
        edge_index_2_torch = torch.empty((2, 0), dtype=torch.long)
        edge_attr_2_torch = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index_2_torch = torch.tensor(edge_index_2, dtype=torch.long).t().contiguous()
        edge_attr_2_torch = torch.tensor(edge_attr_2, dtype=torch.float)
        
    return edge_index_1_torch, edge_attr_1_torch, edge_index_2_torch, edge_attr_2_torch, idx_to_user, idx_to_event


import pandas as pd
import networkx as nx
import torch
from pathlib import Path

def generate_user_features(user_nodes, edge_index_1, edge_attr_1, edge_index_2, edge_attr_2, raw_dir="contagion-seeding-in-meetup/notebooks/data/raw", out_dir="notebooks/data", n=10):
    """
    Ingests raw data and PyG tensors to generate a flat, production-ready feature matrix.
    Aligns raw CSV IDs with namespaced graph nodes (e.g., 'm_2069').
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Initializing feature matrix...")
    # Base dataframe aligned with user_nodes (assumed to be namespaced 'm_...')
    df_features = pd.DataFrame({'member_id': user_nodes}).set_index('member_id')

    # --- 1. Load & Process Metadata ---
    print("Processing spatial metadata...")
    df_members = pd.read_csv(raw_path / "meta-members.csv", usecols=['member_id', 'lat', 'lon'])
    # Namespace and strip .0 artifacts
    df_members['member_id'] = 'm_' + df_members['member_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_members = df_members.set_index('member_id')
    df_features = df_features.join(df_members, how='left')

    # --- 2. Process Group Behavior ---
    print("Processing group membership counts...")
    df_mem_groups = pd.read_csv(raw_path / "member-to-group-edges.csv")
    # Namespace and strip .0 artifacts
    df_mem_groups['member_id'] = 'm_' + df_mem_groups['member_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    group_counts = df_mem_groups.groupby('member_id').size().rename('group_count')
    df_features = df_features.join(group_counts, how='left').fillna({'group_count': 0})

    # --- 3. Process Temporal Event Sequences ---
    print("Extracting chronological event sequences...")
    df_rsvps = pd.read_csv(raw_path / "rsvps.csv", usecols=['member_id', 'event_id'])
    df_events = pd.read_csv(raw_path / "meta-events.csv", usecols=['event_id', 'time'])
    
    # Namespace IDs in both frames before joining
    df_rsvps['member_id'] = 'm_' + df_rsvps['member_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_rsvps['event_id'] = 'e_' + df_rsvps['event_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_events['event_id'] = 'e_' + df_events['event_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Join to get timestamps, sort, and extract the latest n
    df_rsvps_time = df_rsvps.merge(df_events, on='event_id')
    df_rsvps_time['time'] = pd.to_datetime(df_rsvps_time['time'], format='mixed', errors='coerce')
    df_rsvps_time = df_rsvps_time.sort_values(by=['member_id', 'time'], ascending=[True, False])
    
    # Group by member, take top n, and format as comma-separated string
    last_n_events = (
        df_rsvps_time.groupby('member_id')
        .head(n)
        .groupby('member_id')['event_id']
        .apply(lambda x: ','.join(x.astype(str)))
        .rename('last_n_events')
    )
    df_features = df_features.join(last_n_events, how='left')

    # --- 4. Process Network Features ---
    print("Reconstructing 1-simplex graph for topological metrics...")
    G_backbone = nx.Graph()
    G_backbone.add_nodes_from(user_nodes) # user_nodes are already namespaced
    
    # Map contiguous integer indices back to namespaced IDs
    sources_1 = edge_index_1[0].numpy()
    targets_1 = edge_index_1[1].numpy()
    edges_1 = [(user_nodes[s], user_nodes[t]) for s, t in zip(sources_1, targets_1)]
    G_backbone.add_edges_from(edges_1)

    print("Computing Degree, PageRank, and Clustering...")
    df_features['degree_1_simplex'] = pd.Series(dict(G_backbone.degree()))
    df_features['pagerank'] = pd.Series(nx.pagerank(G_backbone, alpha=0.85))
    df_features['clustering_coeff'] = pd.Series(nx.clustering(G_backbone))

    print("Computing 2-simplex participation...")
    if edge_index_2.numel() > 0:
        nodes_in_2_simplices = edge_index_2.flatten().numpy()
        # Map back to namespaced IDs
        mapped_nodes = [user_nodes[n] for n in nodes_in_2_simplices]
        simplicial_counts = pd.Series(mapped_nodes).value_counts().rename('simplicial_degree')
        df_features = df_features.join(simplicial_counts, how='left').fillna({'simplicial_degree': 0})
    else:
        df_features['simplicial_degree'] = 0.0

    # --- 5. Finalize and Export ---
    print("Exporting feature matrix...")
    df_features.to_csv(out_path / "user_features.csv")
    print(f"Successfully generated features for {len(df_features)} users.")
    
    return df_features



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_feature_distributions(file_path="contagion-seeding-in-meetup/notebooks/data/user_features.csv"):
    if not Path(file_path).exists():
        print(f"Error: {file_path} not found. Ensure you have run generate_user_features() first.")
        return

    # Load namespaced features
    df = pd.read_csv(file_path, index_col='member_id')
    
    # 1. Transform event string to sequence length
    # This helps see if we are actually getting the 10 events per user
    df['event_seq_len'] = df['last_n_events'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

    # 2. Define features for analysis
    # Network metrics (Degree, PageRank, Simplicial Degree) are often power-law distributed
    # and require log-scales for meaningful visualization.
    features = [
        ('lat', False), 
        ('lon', False), 
        ('group_count', True), 
        ('event_seq_len', False), 
        ('degree_1_simplex', True), 
        ('pagerank', True), 
        ('clustering_coeff', False), 
        ('simplicial_degree', True)
    ]
    
    # 3. Print Summary Stats (Identifying NaN density)
    print("--- Numerical Feature Summary ---")
    print(df[[f[0] for f in features]].describe().T)
    print("\n--- Missing Values (NaN) Count ---")
    print(df[[f[0] for f in features]].isna().sum())

    # 4. Generate Visualization Grid
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for i, (col, use_log) in enumerate(features):
        data = df[col].dropna()
        
        if use_log:
            # We filter for values > 0 for log-scaling
            sns.histplot(data[data > 0], ax=axes[i], kde=True, color='teal', log_scale=True)
            axes[i].set_title(f"{col} (Log Scale)")
        else:
            sns.histplot(data, ax=axes[i], kde=True, color='coral')
            axes[i].set_title(f"{col}")
            
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("contagion-seeding-in-meetup/notebooks/data/feature_distributions.png")
    print("\nDiagnostic plot saved to: contagion-seeding-in-meetup/notebooks/data/feature_distributions.png")



# %%
# Load bipartite graph and member nodes
G_bipartite, member_nodes = load_bipartite_artifacts(input_dir="contagion-seeding-in-meetup/notebooks/data")
print(f"Loaded bipartite graph with {G_bipartite.number_of_nodes()} nodes and {G_bipartite.number_of_edges()} edges.")

# Process the bipartite graph into PyG tensors
edge_index_1, edge_attr_1, edge_index_2, edge_attr_2, user_idx, event_idx = process_multiplex_graph(
    G_bipartite,
    member_nodes,
    alpha=0.08,
    max_event_size=50
)
print(f"Processed multiplex graph: {edge_index_1.shape[1]} 1-simplices and {edge_index_2.shape[1]//6} 2-simplices.")

# Print to file
torch.save(edge_index_1, "contagion-seeding-in-meetup/notebooks/data/edge_index_simple.pt")
torch.save(edge_attr_1, "contagion-seeding-in-meetup/notebooks/data/edge_attr_simple.pt")
torch.save(edge_index_2, "contagion-seeding-in-meetup/notebooks/data/edge_index_hyper.pt")
torch.save(edge_attr_2, "contagion-seeding-in-meetup/notebooks/data/edge_attr_hyper.pt")

import pickle
with open("contagion-seeding-in-meetup/notebooks/data/user_idx.pkl", "wb") as f:
    pickle.dump(user_idx, f)
with open("contagion-seeding-in-meetup/notebooks/data/event_idx.pkl", "wb") as f:
    pickle.dump(event_idx, f)

# Generate user features and export to CSV
df_user_features = generate_user_features(
    member_nodes, 
    edge_index_1, 
    edge_attr_1, 
    edge_index_2, 
    edge_attr_2, 
    raw_dir="contagion-seeding-in-meetup/notebooks/data/raw", 
    out_dir="contagion-seeding-in-meetup/notebooks/data",
    n=10
)
print("Feature generation complete. Sample of generated features:")
print(df_user_features.head())

# Event features (Stub)
# generate_mock_event_vectors.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def generate_event_vectors(input_pkl="contagion-seeding-in-meetup/notebooks/data/event_idx.pkl", output_csv="contagion-seeding-in-meetup/notebooks/data/event_features.csv", vector_dim=64):
    in_path = Path(input_pkl)
    out_path = Path(output_csv)
    
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        sys.exit(1)

    # Load the deterministic list of event IDs
    with open(in_path, "rb") as f:
        event_idx = pickle.load(f)
        
    num_events = len(event_idx)
    print(f"Loaded {num_events} event IDs.")

    # Generate a dense matrix of random vectors (standard normal distribution)
    print(f"Generating {vector_dim}-dimensional vectors...")
    vectors = np.random.randn(num_events, vector_dim).astype(np.float32)

    # Construct the dataframe
    columns = [f"dim_{i}" for i in range(vector_dim)]
    df = pd.DataFrame(vectors, index=event_idx, columns=columns)
    df.index.name = "event_id"

    # Export to CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"Successfully exported vector matrix to {out_path}.")

generate_event_vectors()

# Analyze feature distributions
analyze_feature_distributions()
    
    