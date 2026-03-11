# Notification Seeding Model — Experiment Plan

## Goal

Develop and evaluate a notification seeding model that selects which users to notify
when a new event is posted on SideQuest. The model optimizes two objectives:

1. **Fill the event**: maximize RSVPs by exploiting social reinforcement (complex contagion)
2. **Preserve residual graph quality**: avoid burning high-centrality users on every event,
   keeping the network's future spreading capacity intact for subsequent posts

We use the **Meetup.com dataset** as a structural analog to SideQuest — local events,
explicit RSVPs, category tags, geographic data — and model social spread as complex
contagion on the co-occurrence graph (a weighted pairwise graph where users are connected
by shared event attendance).

---

## Repository Structure

```
README.md
data/
  raw/                    # downloaded Meetup dataset files
  processed/              # graph objects, feature tensors, splits
notebooks/
  00_eda.ipynb            # schema exploration, distributions
  01_graph_construction.ipynb
  02_contagion_calibration.ipynb
src/
  graph/                  # graph construction + feature extraction
  contagion/              # cascade simulator
  models/                 # all model implementations
  evaluation/             # harness + metrics
results/
  figures/
  tables/
```

---

## Phase 0: Data Acquisition

### 0.1 Download Meetup Dataset

The most accessible version is the Kaggle Meetup dataset dump. The required tables are:

| Table | Key columns |
|-------|------------|
| `members` | `member_id`, `city`, `country`, `joined` |
| `groups` | `group_id`, `category`, `city`, `lat`, `lon`, `created` |
| `events` | `event_id`, `group_id`, `category`, `time`, `created`, `rsvp_limit`, `yes_rsvp_count` |
| `rsvps` | `event_id`, `member_id`, `response` (yes/no), `created` (timestamp) |

The `rsvps.created` timestamp is critical — it gives the ordering of RSVPs within each
event, which is required to compute each user's social exposure at the moment they RSVPed
(how many of their connections had already RSVPed before them).

### 0.2 Exploratory Analysis

- Schema validation and null audit
- RSVP distribution: RSVPs per event (shape, median, tail)
- Temporal range and geographic coverage
- Category distribution
- Check whether a friendship/social graph is present in the version obtained;
  if not, the co-occurrence graph is the only social signal available (same situation as SideQuest)

---

## Phase 1: Graph Construction

### 1.a — Co-occurrence Graph
*Depends on: Phase 0*

Build the social graph from shared event attendance:

1. **Bipartite graph** $B = (U, E_\text{ev}, A)$ — users × events, $A_{ue} = 1$ if user
   $u$ RSVPed yes to event $e$
2. **Pairwise co-occurrence graph**: weighted graph $G = (U, E)$ where an edge $(u,v)$
   exists if $u$ and $v$ both RSVPed yes to at least one common event. Edge weight with
   recency decay:
   $$w(u,v) = \sum_{e \in \text{shared}(u,v)} \exp\!\left(-\alpha \cdot \Delta t_e\right)$$
   where $\Delta t_e$ is the age of event $e$ in days. Use $\alpha$ such that an event
   6 months old has ~50% weight.

**Output**: adjacency list, edge weight matrix.

### 1.b — Node Feature Extraction
*Depends on: Phase 0 — can run in parallel with 1.a*

**Structural features** (computed from pairwise projection once 1.a is done):
- Degree $d(u)$
- Local clustering coefficient $C(u)$
- Betweenness centrality — approximate via random pivot sampling for large graphs
- Ego network density

**Behavioral features** (from RSVP history, requires only bipartite graph):
- Total yes-RSVP count
- Category affinity vector (fraction of RSVPs per category, softmax-normalized)
- Average event size attended (proxy for preference for large vs intimate events)
- Temporal activity pattern: preferred days of week, time of day

**Content embedding**:
- Per-user embedding: mean of `all-MiniLM-L6-v2` embeddings of attended event
  titles/descriptions. Matches SideQuest's `UserInterestEmbedding` infrastructure
  (384-dim, same model).

---

## Phase 2: Susceptibility Prior & Contagion Calibration

### 2.a — Network-Unaware Susceptibility Model (BCE Prior)
**Goal:** Establish the independent probability of user $u$ joining event $e$, completely isolated from graph propagation dynamics. This acts as the base susceptibility prior for the complex contagion simulation.

**Architecture & Training:**
1.  **Input Features:** 
    *   User vector: Concatenation of historical RSVP frequency, categorical category affinities (One-Hot Encoded), and demographic data.
    *   Event vector: Event category, organizer ID, time of day/week, geographic coordinates.
2.  **Embedding:** Learn separate embeddings for User and Event vectors using a 2-layer Multi-Layer Perceptron (MLP).
3.  **Forward Pass:** Calculate the dot product of the User and Event embeddings, passed through a sigmoid activation to output $P_{indep}(u, e) \in (0, 1)$.
4.  **Loss Function:** Binary Cross-Entropy (BCE) trained strictly on historical, temporally split RSVP logs (1 = attended, 0 = exposed but did not attend).

### 2.b — Calibrated Complex Contagion Simulation Schema
**Goal:** Integrate the prior $P_{indep}(u, e)$ into the complex contagion formulation to simulate realistic cascade dynamics on the final 10% chronological holdout set.

**Mathematical Schema:**
Standard complex contagion assumes uniform node susceptibility. We modify this to be highly heterogeneous.
Let $k$ be the number of user $u$'s neighbors who have already RSVP'd to event $e$.
Let $\beta_1$ be the baseline pairwise transmission rate.
Let $\beta_2$ be the social reinforcement rate triggered at a threshold (e.g., $k \ge 2$).

The modified probability that user $u$ RSVPs at time step $t$, $P_{join}(u, e | k)$, is defined as the union of independent attendance and socially driven attendance:

`P_join(u, e | k) = P_indep(u, e) + (1 - P_indep(u, e)) * [ 1 - (1 - beta_1)^k + beta_2 * I[k >= 2] ]`

*Note: $\mathbb{I}$ is the indicator function. The parameters $\beta_1$ and $\beta_2$ are empirically derived via grid search to minimize the distributional distance (Wasserstein distance) between simulated cascade sizes and historical cascade sizes in the validation set.*

### 2.d — Cascade Simulation Execution Mechanics
**Goal:** Define the discrete time-step mechanics of how a cascade unfolds once a model has selected a seed set $S$. This simulator acts as the environment for both imitation learning pretraining, DRL reward calculation, and final evaluation.

**Simulation Algorithm:**
The simulation follows a discrete-time, monotonic Susceptible-Infected (SI) contagion model on the pairwise co-occurrence graph. Once a user RSVPs (becomes Infected), they remain in that state and continuously exert social pressure on their susceptible neighbors.

1.  **Initialization ($t=0$):** 
    *   The model outputs a seed set of size $K$.
    *   The active RSVP set is initialized as $I_0 = S$.
    *   The susceptible pool is defined as $V_{sus} = V \setminus I_0$.
2.  **Propagation Step ($t$ to $t+1$):**
    *   Identify the candidate frontier: any susceptible user $u \in V_{sus}$ who shares an edge with at least one user in $I_t$.
    *   For each candidate $u$, calculate their current social exposure: $k_u = | \mathcal{N}(u) \cap I_t |$.
    *   Calculate their dynamic infection probability $P_{join}(u, e \mid k_u)$ using the calibrated schema defined in 2.b.
3.  **Activation Sampling:**
    *   For each candidate $u$, sample a random float $x \sim \text{Uniform}(0, 1)$.
    *   If $x < P_{join}(u, e \mid k_u)$, the user RSVPs. They are added to the newly infected set $\Delta I$.
4.  **State Update:**
    *   $I_{t+1} = I_t \cup \Delta I$.
    *   $V_{sus} = V_{sus} \setminus \Delta I$.
5.  **Termination Condition:**
    *   The simulation advances to the next time step until either $\Delta I = \emptyset$ (no new RSVPs in the current step) or $|I_{t+1}|$ reaches the event's hard capacity limit. 
    *   The final cascade size is $|I_{final}|$.

### 2.d — Data Preparation Pipeline
1.  **Graph Tensors:** Construct bipartite (user-event) and homogeneous (user-user via co-attendance) adjacency matrices.
2.  **Edge Weights:** Implement a time-decay factor for user-user edges. Edge weight $w_{u,v} = \sum \exp(-\lambda \cdot \Delta t)$, where $\Delta t$ is days since co-attendance.
3.  **Temporal Leakage Prevention:** Split events strictly chronologically (e.g., Train: 2019-2021, Val: Jan-Jun 2022, Test: Jul-Dec 2022). The homogeneous graph at time $T$ must only contain edges formed prior to $T$.

---

# Phase 3: Model Development

All sub-phases can be developed in parallel once Phase 2 is complete. 

### Universal Two-Stage Training Regime
To rigorously evaluate the impact of training methodology versus architectural complexity, **all machine learning models (M1, M2, M3, M4)** will be trained using a standardized two-stage pipeline. Every model operates autoregressively, maintaining an internal context of the already-selected seed set $S_t$.

*   **Stage 1: Imitation Learning Pretraining (IL)**
    *   **Goal:** Bootstrapping the models to approximate an exhaustive greedy search of the simulation environment.
    *   **Process:** For a given event and partial seed set $S_t$, the calibrated simulator (Phase 2.b) tests *every* valid candidate node $v \notin S_t$ and records the expected cascade size $C_v$. 
    *   **Optimization:** We apply a softmax over all $C_v$ to create a target probability distribution. All models are trained via cross-entropy (or KL divergence) to align their predicted selection scores with this optimal target distribution.

*   **Stage 2: Deep Reinforcement Learning (DRL)**
    *   **Goal:** Transition the pretrained models into an interactive environment to optimize for the dual objective (cascade spread vs. residual network health). We will use Proximal Policy Optimization (PPO) or DQN depending on the model's stability.
    *   **MDP Formulation:**
        *   **State ($s_t$):** $(G, S_t, e)$ — graph state, current seed set, event context.
        *   **Action ($a_t$):** Add user $u \notin S_t$ to $S_t$.
        *   **Reward ($r_t$):** $\Delta$ cascade RSVPs from adding $u$ minus a penalty for selecting nodes with high centrality or recent notification history:
            $$r_t = \left[ \text{CascadeSize}(S_t \cup \{u\}) - \text{CascadeSize}(S_t) \right] - \alpha \cdot \text{BurnoutPenalty}(u)$$
    *   **Optimization:** The models from Stage 1 are fine-tuned via RL gradients to maximize expected cumulative reward until the budget $K$ is exhausted.

---

### 3.a — Baselines (B1, B2, B3)
*Depends on: Phase 2.b — no ML training required.*
Implement first. These set the performance floor and are required reference points.

*   **B1 — Random Seeding:** Select $K$ users uniformly at random from the notification-eligible pool.
*   **B2 — Degree Centrality Greedy:** Score eligible users by degree $d(u)$ in the co-occurrence graph, notify top-$K$. Represents the intuitive default: "always notify the most connected people." Fast, but destroys residual graph quality.
*   **B3 — CELF Greedy (Simple IC):** The Leskovec et al. (2007) Cost-Effective Lazy Forward greedy algorithm, achieving a $(1-1/e) \approx 0.63$ approximation of the optimal seed set under the Independent Cascade model. The theoretical gold standard for simple-contagion maximization.

### 3.b — Pointwise MLP Autoregressive Ranker (M1)
*No graph structure — establishes the value-add of graph awareness.*

*   **Architecture:** 3-layer MLP on concatenated scalar features:
    *   $\cos(z_u, z_e)$ — cosine similarity between user and event embeddings.
    *   Geographic distance $d(u,e)$.
    *   Category match: dot product of user category affinity vector and event category one-hot.
    *   User historical RSVP rate.
    *   Event recency, current RSVP count, event size limit.
    *   **Autoregressive Context:** A dynamically updated vector representing the aggregated scalar features of the selected seed set $S_t$.
*   **Seeding:** Iteratively score all eligible users against the updated context, select the top-$1$, append to $S_t$, update context, and repeat until budget $K$.

### 3.c — LightGCN Collaborative Filter (M2)
*Uses the bipartite attendance graph only — answers if knowing who attends similar events helps without explicit social graph structure.*

*   **Architecture:** LightGCN (He et al. 2020) on the bipartite user-event graph. Propagation averages neighbor embeddings without non-linearity:
    $$e_u^{(k+1)} = \sum_{e \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(e)|}} e_e^{(k)}, \quad e_e^{(k+1)} = \sum_{u \in \mathcal{N}(e)} \frac{1}{\sqrt{|\mathcal{N}(e)||\mathcal{N}(u)|}} e_u^{(k)}$$
    Final user/event embeddings are the mean across $K$ propagation layers.
*   **Seeding (Autoregressive):** The model scores nodes via dot products. To maintain autoregression, candidates sharing high bipartite overlap with $S_t$ receive a dynamically calculated structural penalty, forcing the model to explore distinct event-clusters.

### 3.d — GAT Dual-Head Scorer (M3)
*Operates on the pairwise co-occurrence graph with the event embedding as a global conditioning signal.*

*   **Architecture:**
    *   **Node features:** Concatenation of user embedding (384-dim), structural features (degree, betweenness, clustering), behavioral features, and cooldown state.
    *   **Graph encoder:** 3-layer GAT (Veličković et al. 2018), 8 attention heads per layer. Edge features injected as attention biases.
    *   **Event cross-attention:** After GAT encoding, each node embedding is cross-attended with the event embedding $z_e$.
    *   **Outputs:** An RSVP probability head and a Structural Value head (predicting network health contributions).
*   **Seeding (Greedy Autoregressive):** 
    1. Score eligible users: $s_u = \alpha \cdot \hat{P}(\text{RSVP}_u) + (1-\alpha) \cdot \hat{V}_u$
    2. Select $u^* = \arg\max_u s_u$, add to seed set.
    3. **Autoregressive step:** Downweight 1-hop neighbors of $u^*$ by their edge weight to $u^*$ (suppressing overlapping influence).

### 3.e — S2V Sequential Seeder (M4)
*Unlike M3, which scores nodes and then applies an external neighbor-suppression heuristic, M4 explicitly passes the selected set $S_t$ into the graph convolutions to learn the combinatorial value natively.*

*   **Architecture:** Based on structure2vec (Dai et al. 2017).
    $$\mu_u^{(t+1)} = \text{ReLU}\left(\theta_1 x_u + \theta_2 \sum_{v \in \mathcal{N}(u)} w(u,v)\mu_v^{(t)} + \theta_3 \sum_{v \in S_t} \mu_v^{(t)}\right)$$
    The $\theta_3$ term natively encodes which users are already seeded, allowing the network to internally learn decreasing marginal returns and natural diversity without external heuristics.
*   **Seeding:** The Q-function scores candidate additions dynamically based on the S2V embeddings at step $t$.

---

# Phase 4: Comprehensive Evaluation Framework

### 4.a — Monte Carlo Evaluation Harness
**Goal:** Standardize how the simulation is executed across the test holdout set to ensure fair, reproducible benchmarking for the Results Matrix.

Because the contagion simulation relies on probabilistic activation sampling, a single run is highly subject to variance. To evaluate the true expected value of a model's seeding policy, we must run a Monte Carlo simulation for every event.

**Evaluation Loop:**
For every event $e$ in the final **10% chronological test holdout set**:
1.  **Graph Snapshot Loading:** Load $G_e$, which contains only nodes and edges that existed strictly prior to the timestamp of event $e$.
2.  **Seed Selection:** The model under evaluation processes $G_e$ and event features to output its optimal seed set $S_e$.
3.  **Monte Carlo Trials:**
    *   Instantiate the simulation (defined in 2.d) using $S_e$.
    *   Execute $M=1000$ independent trials of the simulation to completion.
    *   Record the cascade size for each trial: $C_1, C_2, \dots, C_{1000}$.
4.  **Event-Level Metric Aggregation:**
    *   Calculate Expected Success: $\mathbb{E}[|C|] = \frac{1}{1000} \sum C_m$.
    *   Calculate Expected Efficiency: $\eta = \mathbb{E}[|C|] / K$.
5.  **Residual Health Calculation:**
    *   Simulate the "burnout" by removing $S_e$ from the graph snapshot.
    *   Calculate $\Delta\lambda_2 = \lambda_2(G_e) - \lambda_2(G_e \setminus S_e)$ using power iteration.
6.  **Global Aggregation:** Average the expected success, efficiency, and health metrics across all test events to populate the final row for this model in the 5.a Results Matrix.

### 4.b — Benchmark Run and Results Matrix
Run all models on the full test event set (last 10% chronological holdout). Because all ML models are subjected to the two-stage training regime, the final evaluation matrix will explicitly break out performance by training stage (Imitation vs. DRL).

*Metrics tracked:*
*   **Success:** Mean simulated cascade size (RSVPs).
*   **Health:** $\Delta\lambda_2$ (Algebraic Connectivity Fiedler value). A smaller drop indicates better preservation of residual network capacity.
*   **Efficiency:** Notifications sent per RSVP gained.
*   **RL Objective:** Mean episode reward achieved in the simulator.
*   **Runtime:** Wall-clock inference runtime per event in milliseconds.

| Model & Training Phase | Mean RSVPs ↑ | $\Delta\lambda_2$ ↓ | Notifs / RSVP ↓ | Mean Episode Reward ↑ | Inference Runtime (ms) ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 Random** | | | | N/A | |
| **B2 Degree Centrality** | | | | N/A | |
| **B3 CELF-IC** | | | | N/A | |
| **M1 MLP (IL Only)** | | | | | |
| **M1 MLP (+ DRL)** | | | | | |
| **M2 LightGCN (IL Only)** | | | | | |
| **M2 LightGCN (+ DRL)**| | | | | |
| **M3 GAT (IL Only)** | | | | | |
| **M3 GAT (+ DRL)** | | | | | |
| **M4 S2V (IL Only)** | | | | | |
| **M4 S2V (+ DRL)** | | | | | |

---

## Appendix: Key References

- Centola & Macy (2007) — "Complex Contagions and the Weakness of Long Ties" (*American Journal of Sociology*) — foundational complex contagion theory
- Iacopini et al. (2019) — "Simplicial Models of Social Contagion" (*Nature Communications*) — formal model for social reinforcement (basis for $\beta_2$ term)
- Dai et al. (2017) — "Learning Combinatorial Optimization Algorithms over Graphs" (S2V-DQN)
- Veličković et al. (2018) — "Graph Attention Networks"
- He et al. (2020) — "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
- Leskovec et al. (2007) — "Cost-effective Outbreak Detection in Networks" (CELF)
