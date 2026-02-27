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
notification_model_plan.md
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

## Phase 2: Calibration and Data Preparation

### 2.a — Complex Contagion Calibration
*Depends on: Phase 1.a*
*Can run in parallel with: 2.b*

**Goal**: empirically estimate $\beta_1$ (pairwise infection rate) and $\beta_2$
(social reinforcement) from observed RSVP cascades, rather than assuming them.

**Method**:

1. For each (user $u$, event $e$) pair where $u$ RSVPed yes, compute the **social
   exposure at RSVP time**: the number of $u$'s co-occurrence graph neighbors who had
   already RSVPed yes to $e$ before $u$ did (using `rsvps.created` ordering).
2. Bin by social exposure count $k = 0, 1, 2, 3, 4+$.
3. Compute empirical $P(\text{RSVP yes} \mid \text{exposure} = k)$ for each bin.
4. Fit the complex contagion curve:
   $$P(k) = 1 - (1 - \beta_1)^k + \beta_2 \cdot \mathbf{1}[k \geq 2]$$
   or a more general form: $P(k) = \sigma(a + b \cdot k + c \cdot \mathbf{1}[k \geq 2])$
5. Extract $\hat\beta_1, \hat\beta_2$ from the fit.

**Validation**:
- If $P(k)$ is superlinear in $k$, complex contagion is confirmed in the data.
- If linear or sublinear, simple IC is sufficient and $\beta_2 \approx 0$.
- Report $\hat\beta_2 / \hat\beta_1$ as the "social reinforcement multiplier".

**Output**: calibrated $(\hat\beta_1, \hat\beta_2)$ for the cascade simulator in Phase 4.

### 2.b — ML Data Preparation
*Depends on: Phase 1.a, 1.b*
*Can run in parallel with: 2.a*

1. **Temporal train/val/test split** — split events by time (70/15/15). Never random
   split — temporal ordering prevents leakage (a model seeing future RSVPs would have
   inflated performance on past events).
2. **Negative sampling** — for each positive (user, event) RSVP pair in training data,
   sample $k$ negative pairs (user was in the geographic radius of the event but did not
   RSVP). Negatives should be plausible non-attendees, not random users.
3. **Feature matrix** — assemble node features from 1.b into tensors.
4. **Graph snapshots** — for each held-out event $e_t$, construct graph $G_t$ using only
   RSVPs from events that were posted strictly before $t$. This is the exact graph state
   a seeding model would observe at event-post time.

---

## Phase 3: Model Development

*All sub-phases can be developed in parallel once Phase 2 is complete.*
*Phase 3.e additionally depends on Phase 2.a for the cascade reward signal.*

### 3.a — Baselines (B1, B2, B3)
*Depends on: Phase 2.b — no ML training required*

Implement first. These set the performance floor and are required reference points.

**B1 — Random Seeding**

Select $K$ users uniformly at random from the notification-eligible pool (within
distance threshold, not in cooldown, preferences allow). True floor; should be beaten
by everything else.

**B2 — Degree Centrality Greedy**

Score eligible users by degree $d(u)$ in the co-occurrence graph, notify top-$K$.
Represents the intuitive default: "always notify the most connected people." Fast,
requires no training, but ignores event-user fit and destroys residual graph quality
by systematically burning hubs.

**B3 — CELF Greedy (Simple IC)**

The Leskovec et al. (2007) Cost-Effective Lazy Forward greedy algorithm, which
achieves a $(1 - 1/e) \approx 0.63$ approximation of the optimal seed set under the
Independent Cascade model. Edge propagation probabilities set proportional to
normalized co-occurrence weight. This is the theoretical gold standard for
simple-contagion influence maximization — all DL models should be compared against it.
The gap between CELF and DL models quantifies the value of (a) complex contagion
awareness and (b) personalized event-user matching.

---

### 3.b — Pointwise MLP RSVP Ranker (M1)
*Depends on: Phase 2.b*
*No graph structure — establishes the value-add of graph awareness*

**Architecture**: 3-layer MLP on concatenated scalar features:
- $\cos(z_u, z_e)$ — cosine similarity between user embedding and event embedding
- Geographic distance $d(u, e)$
- Category match: dot product of user category affinity vector and event category one-hot
- User historical RSVP rate (total yes RSVPs / total events exposed to)
- Event recency, current RSVP count, event size limit

**Training**: binary cross-entropy on (user, event, RSVP=0/1) pairs.

**Seeding**: score all eligible users, select top-$K$.

**Why include this**: if M1 matches the GNN models, graph structure isn't adding
measurable value and a simpler production system is preferable. This is the baseline
that proves or disproves the need for graph-aware models.

---

### 3.c — LightGCN Collaborative Filter (M2)
*Depends on: Phase 2.b*

**Architecture**: LightGCN (He et al. 2020) on the bipartite user-event graph.
Propagation averages neighbor embeddings without non-linearity or feature transformation:
$$e_u^{(k+1)} = \sum_{e \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(e)|}} e_e^{(k)}, \quad
e_e^{(k+1)} = \sum_{u \in \mathcal{N}(e)} \frac{1}{\sqrt{|\mathcal{N}(e)||\mathcal{N}(u)|}} e_u^{(k)}$$

Final user/event embeddings: mean across $K$ propagation layers. Score (user, event)
pair by dot product of final embeddings.

**Training**: Bayesian Personalized Ranking (BPR) loss on (user, positive event,
negative event) triples.

**Seeding**: score new event $e_\text{new}$ against all eligible users, select top-$K$.

**Why include this**: standard collaborative filtering GNN baseline. Uses the bipartite
attendance graph only — no co-occurrence graph, no structural features, no event
cross-attention. Answers: "does knowing who attends similar events help, even without
explicit social graph structure?"

---

### 3.d — GAT Dual-Head Scorer (M3)
*Depends on: Phase 2.b*

The primary graph-aware model. Operates on the pairwise co-occurrence graph with the
new event's embedding as a global conditioning signal.

**Architecture**:

1. **Node features**: concatenation of user embedding (384-dim), structural features
   (degree, betweenness, clustering coefficient), behavioral features (category affinity,
   RSVP rate), cooldown state (days since last notification).
2. **Graph encoder**: 3-layer GAT (Veličković et al. 2018), 8 attention heads per layer.
   Edge features (co-occurrence weight, recency-weighted weight) injected as attention
   biases.
3. **Event cross-attention**: after GAT encoding, each node embedding is cross-attended
   with the event embedding $z_e$ (event as query, nodes as keys/values). This
   conditions scores on the specific event being promoted rather than producing
   context-free user rankings.
4. **Two output heads**:
   - **RSVP head**: $\hat P(\text{RSVP} \mid u \text{ notified}, e)$ — trained
     supervised on RSVP labels via binary cross-entropy.
   - **Structural value head**: $\hat V(u \mid G)$ — trained with self-supervised
     targets (normalized betweenness, local algebraic connectivity contribution)
     during pre-training, then updated via reward signal.

**Selection**: greedy with neighbor suppression:
- Score eligible users: $s_u = \alpha \cdot \hat P(\text{RSVP}_u) + (1-\alpha) \cdot \hat V_u$
- Select $u^* = \arg\max_u s_u$, add to seed set
- Downweight 1-hop neighbors of $u^*$ by their edge weight to $u^*$ (they will likely
  hear about the event organically through the already-selected user)
- Repeat until budget $K$ is exhausted

**Ablations to run** (see Phase 5.d): RSVP head only; no neighbor suppression;
no event cross-attention; no structural value head.

---

### 3.e — S2V-DQN Sequential Seeder (M4)
*Depends on: Phase 2.b, Phase 2.a (cascade simulator as reward during training)*

A reinforcement learning agent that selects seed users one at a time, learning the value
of each addition *given the users already selected*. Based on structure2vec-DQN
(Dai et al. 2017).

Unlike M3 which scores users independently then applies a greedy heuristic, M4
learns the *combinatorial* value of a seed set — it explicitly represents "I already
selected A, so selecting A's close co-attendee B is now less valuable."

**MDP formulation**:
- **State** at step $t$: $(G, S_t, e)$ — graph with node features, current seed set,
  event context
- **Action**: add user $u \notin S_t$ to $S_t$
- **Reward**: $\Delta$ cascade RSVPs from adding $u$, estimated via the calibrated
  complex contagion simulator (Monte Carlo, $M=50$ samples per step during training)
- **Episode terminates** when $|S_t| = K$ (budget exhausted) or event capacity would
  be saturated

**Network**: structure2vec message passing with explicit selected-set context:
$$\mu_u^{(t+1)} = \text{ReLU}\!\left(\theta_1 x_u + \theta_2 \sum_{v \in N(u)} w(u,v)\,\mu_v^{(t)} + \theta_3 \sum_{v \in S_t} \mu_v^{(t)}\right)$$

The $\theta_3$ term is the key difference — it encodes which users are already seeded
so the network learns decreasing marginal returns and natural diversity. Q-function:
$$Q(S_t, u) = \theta_4^\top \cdot \text{ReLU}\!\left(\left[\theta_5 \cdot \sum_{v \in V} \mu_v,\; \theta_6 \cdot \mu_u\right]\right)$$

**Training**: DQN with experience replay and target network. Warm-start from M3 RSVP
head predictions to avoid cold exploration.

**Why include this**: only model that optimizes directly for cascade spread under the
calibrated complex contagion model. The gap between M4 and M3 quantifies the value of
learned sequential decision-making over independent scoring + greedy selection.

---

## Phase 4: Evaluation Framework

*Can be developed in parallel with Phase 3. Depends on Phase 2.a for contagion parameters.*

### 4.a — Monte Carlo Cascade Simulator
*Depends on: Phase 2.a*

Implements the calibrated complex contagion spread on the pairwise co-occurrence graph:

1. Initialize infected set $I = S$ (seed set)
2. For each susceptible node $u$, compute social exposure:
   $$k_u = |\{v \in N(u) : v \in I\}|$$
3. Infection probability using calibrated parameters:
   $$P(\text{infect}\,u) = 1 - (1-\hat\beta_1)^{k_u} + \hat\beta_2 \cdot \mathbf{1}[k_u \geq 2]$$
4. Sample infections; add newly infected to $I$; repeat until no new infections
   or event capacity reached
5. Run $M = 1000$ independent trials; report mean ± std RSVPs

**Sensitivity**: also run with $\beta_2 = 0$ (simple IC) to quantify the contribution
of complex contagion dynamics to the results.

### 4.b — Residual Graph Quality Metrics

After selecting seed set $S$ (which enters cooldown), compute quality of
$G' = G \setminus S$:

| Metric | Interpretation | Compute method |
|--------|---------------|----------------|
| $\Delta\lambda_2$ | Change in algebraic connectivity (Fiedler value) of graph Laplacian | Power iteration on $L(G')$ |
| 2-hop coverage | Fraction of $V \setminus S$ reachable within 2 hops from $V \setminus S$ | BFS |
| $\Delta R_\text{eff}$ | Change in sum of pairwise effective resistances | Johnson-Lindenstrauss random projection estimate |

Report all three; $\Delta\lambda_2$ is the primary residual quality metric.

### 4.c — Evaluation Harness

For each held-out event $e_t$ in the test split:
1. Construct graph snapshot $G_t$ (RSVPs prior to $t$ only)
2. Apply each model to select seed set $S$ of size $K$
3. Run cascade simulator from $S$ on $G_t$
4. Compute residual graph metrics for $G_t$ after removing $S$ from eligible pool
5. Record: cascade RSVPs, notifications-per-RSVP, $\Delta\lambda_2$, coverage,
   wall-clock runtime

---

## Phase 5: Comparative Evaluation

*Depends on: Phase 3 + Phase 4*

### 5.a — Benchmark Run

Run all 7 models on the full test event set.

| Model | Mean RSVPs ↑ | Notifs/RSVP ↓ | $\Delta\lambda_2$ ↑ | 2-hop coverage ↑ | Runtime |
|-------|------------|----------------|---------------------|-----------------|---------|
| B1 Random | | | | | |
| B2 Degree | | | | | |
| B3 CELF-IC | | | | | |
| M1 MLP | | | | | |
| M2 LightGCN | | | | | |
| M3 GAT | | | | | |
| M4 S2V-DQN | | | | | |

### 5.b — Statistical Testing

Bootstrap confidence intervals on mean cascade RSVPs. Pairwise Wilcoxon signed-rank
tests between all model pairs, Holm-Bonferroni corrected for multiple comparisons.
Identify which improvements are statistically significant vs. noise.

### 5.c — Sensitivity Analysis

Run all models under these sweeps:

- **Budget $K$**: 5, 10, 25, 50, 100 — does model ranking change at different budget sizes?
- **$\beta_2 / \beta_1$ ratio**: 0 (simple IC), 0.5, 1.0, 2.0, 5.0 — how sensitive are
  conclusions to the assumed strength of social reinforcement?
- **Cooldown period $C$**: 1 day, 3 days, 7 days, 14 days — how does tighter rate
  limiting change which strategy wins?
- **Graph density**: bin test events by local subgraph density around the event location;
  measure model performance per bin — do GNN models especially benefit on
  tightly-clustered graphs?

### 5.d — Ablations (M3 GAT)

| Ablation | What it tests |
|----------|--------------|
| RSVP head only (remove $\hat V$) | Does optimizing for residual quality come at a cascade cost? |
| No neighbor suppression | Does explicit diversity in selection help beyond what the graph encoder learns? |
| No event cross-attention | Does conditioning on event features matter vs. context-free user rankings? |
| No structural features (degree, betweenness, clustering) | How much do precomputed graph statistics contribute beyond GNN-learned structure? |

---

## Phase 6: Production Mapping

*Depends on: Phase 5 — apply to the best-performing model(s)*

### 6.1 Schema Translation

| Meetup | SideQuest |
|--------|-----------|
| `rsvps` (yes response) | `EventInteraction` (status=RSVP) |
| `members` | `User` |
| `events` | `Activity` |
| Group membership (co-occurrence proxy) | Co-occurrence from `EventInteraction` |
| Friendship graph (if available) | Not present — co-occurrence only |
| Event category | `Activity.category` |

Note: SideQuest has no explicit friendship graph; only co-occurrence. This is the same
constraint we operate under in experiments since Meetup friendship data may not be
present in the Kaggle dump.

### 6.2 Infrastructure Requirements

- **Graph construction**: nightly batch job rebuilding the co-occurrence adjacency from
  `EventInteraction` table. Store as sparse adjacency in Redis or as a serialized
  NetworkX/PyG graph.
- **Node feature store**: extend or supplement `UserInterestEmbedding` with a parallel
  table storing structural features (degree, betweenness, clustering — precomputed
  nightly via the same batch job).
- **`NotificationLog` table**: track `(user_id, activity_id, sent_at, opened_at,
  rsvp_at)` — this is the essential training signal for eventual fine-tuning on
  SideQuest production data. Cannot train on SideQuest data without it.
- **Online serving**: pre-compute and cache node embeddings from GAT encoder nightly.
  At event-post time, run only the lightweight scoring head (not the full GAT forward
  pass) against the cached embeddings. Target latency < 500ms for seed set selection.
- **Notification preference filter**: apply `event_notif_mode`, `event_notif_categories`,
  and cooldown check before passing candidates to the model — hard constraints,
  not soft penalties.

### 6.3 Path to Production

1. Deploy B3 (CELF) or M1 (MLP ranker) first — low risk, no graph inference latency
2. Instrument `NotificationLog` immediately on first deploy
3. After 4–8 weeks of production data, fine-tune the winning model from Phase 5 on
   real SideQuest (user, event, notified, RSVP) tuples
4. A/B test fine-tuned model vs. baseline; use cascade RSVPs per notification as the
   primary metric (same metric as experiments)

---

## Appendix: Key References

- Centola & Macy (2007) — "Complex Contagions and the Weakness of Long Ties" (*American Journal of Sociology*) — foundational complex contagion theory
- Iacopini et al. (2019) — "Simplicial Models of Social Contagion" (*Nature Communications*) — formal model for social reinforcement (basis for $\beta_2$ term)
- Dai et al. (2017) — "Learning Combinatorial Optimization Algorithms over Graphs" (S2V-DQN)
- Veličković et al. (2018) — "Graph Attention Networks"
- He et al. (2020) — "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
- Leskovec et al. (2007) — "Cost-effective Outbreak Detection in Networks" (CELF)
