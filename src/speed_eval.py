"""
Speed-of-infection evaluation harness.

Given a trained seeding model (or a non-ML baseline), this module measures how
fast a contagion spreads under the SCM model by scoring the timesteps at which
the first `target` post-seed infections occur with an exponential decay:

    speed = E_trials[ sum_{i=1..target} exp(-decay * t_i) ]

`SpeedEvaluator.run_sweep` then sweeps one of lam / lam_d while holding the
other fixed and returns per-model score arrays ready for plotting.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from preprocess import VectorizedSCMSimulator


class SpeedSimulator(VectorizedSCMSimulator):
    """Adds an exponential-decay weighted speed score on top of VectorizedSCMSimulator."""

    def speed_score(
        self,
        initial_infected: Sequence[int],
        target: int = 10,
        decay: float = 0.3,
        max_steps: int = 50,
    ) -> float:
        state = torch.zeros((self.M, self.N), dtype=torch.bool, device=self.device)
        state[:, list(initial_infected)] = True

        score = torch.zeros(self.M, dtype=torch.float, device=self.device)
        remaining = torch.full((self.M,), target, dtype=torch.long, device=self.device)

        for t in range(1, max_steps + 1):
            if (remaining == 0).all():
                break

            float_state = state.float()
            m = torch.sparse.mm(self.A_1, float_state.t()).t()

            n = torch.zeros((self.M, self.N), dtype=torch.float, device=self.device)
            if self.tri_i.numel() > 0:
                for mi in range(self.M):
                    ij = state[mi, self.tri_j]
                    ik = state[mi, self.tri_k]
                    n[mi].scatter_add_(0, self.tri_i, (ij & ik).float())

            prob_surv_1 = torch.pow(1.0 - self.beta.unsqueeze(0), m)
            prob_surv_2 = torch.pow(1.0 - self.beta_delta.unsqueeze(0), n)
            p_inf = 1.0 - prob_surv_1 * prob_surv_2
            p_inf[state] = 0.0

            rolls = torch.rand((self.M, self.N), device=self.device)
            new_infections = rolls < p_inf
            state = state | new_infections

            if not new_infections.any():
                break

            new_counts = new_infections.sum(dim=1).long()
            counted = torch.minimum(new_counts, remaining)
            score += counted.float() * float(np.exp(-decay * t))
            remaining = remaining - counted

        return float(score.mean().item())


# ---------------------------------------------------------------------------
# Seed selectors
# ---------------------------------------------------------------------------


class IterativeModelSeeder:
    """
    Iteratively picks k seeds from a trained imitation model.

    On each step, rebuilds x_batch = [seed_mask | x_static], runs a forward pass,
    masks out already-selected seeds, and argmaxes the remaining logits. This
    lets the model express synergies between successive seeds instead of
    committing to a one-shot top-k ranking.
    """

    def __init__(self, model, static_graph: Dict[str, torch.Tensor], device: str = "cpu"):
        self.model = model
        self.static_graph = static_graph
        self.device = device

    @torch.no_grad()
    def select(
        self,
        event_feat: torch.Tensor,
        k: int,
        initial_seeds: Optional[Sequence[int]] = None,
    ) -> List[int]:
        self.model.eval()
        x_static = self.static_graph["x_static"]
        N = x_static.shape[0]

        seeds: List[int] = list(initial_seeds or [])
        seed_mask = torch.zeros((N, 1), dtype=torch.float, device=self.device)
        if seeds:
            seed_mask[seeds, 0] = 1.0

        event_feat_batch = event_feat.to(self.device).unsqueeze(0)

        for _ in range(k):
            x_batch = torch.cat([seed_mask, x_static], dim=-1).unsqueeze(0)
            logits = self.model(x_batch, self.static_graph, event_feat_batch)
            logits = logits.squeeze(0).clone()
            if seeds:
                logits[seeds] = float("-inf")
            next_seed = int(torch.argmax(logits).item())
            seeds.append(next_seed)
            seed_mask[next_seed, 0] = 1.0

        return seeds


class SimplicialSeederAdapter:
    """Iterative wrapper around `SimplicialSeeder` for fair comparison."""

    def __init__(self, seeder, beta: float = 0.1, beta_delta: float = 0.2):
        self.seeder = seeder
        self.beta = beta
        self.beta_delta = beta_delta

    def select(
        self,
        event_feat: torch.Tensor,
        k: int,
        initial_seeds: Optional[Sequence[int]] = None,
    ) -> List[int]:
        seeds: List[int] = list(initial_seeds or [])
        for _ in range(k):
            candidates = self.seeder(seeds, beta=self.beta, beta_delta=self.beta_delta)
            if not candidates:
                break
            seeds.append(int(candidates[0][0]))
        return seeds


class DegreeSeeder:
    """Pick top-k nodes by simple-graph degree (static, event-agnostic)."""

    def __init__(self, links: List[List[int]]):
        self.order = np.argsort(-np.array([len(nbrs) for nbrs in links])).tolist()

    def select(
        self,
        event_feat: torch.Tensor,
        k: int,
        initial_seeds: Optional[Sequence[int]] = None,
    ) -> List[int]:
        seeds: List[int] = list(initial_seeds or [])
        seen = set(seeds)
        added = 0
        for node in self.order:
            if added == k:
                break
            if node not in seen:
                seeds.append(int(node))
                seen.add(int(node))
                added += 1
        return seeds


class RandomSeeder:
    def __init__(self, num_nodes: int, rng: Optional[np.random.Generator] = None):
        self.N = num_nodes
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def select(
        self,
        event_feat: torch.Tensor,
        k: int,
        initial_seeds: Optional[Sequence[int]] = None,
    ) -> List[int]:
        seeds: List[int] = list(initial_seeds or [])
        pool = np.array([i for i in range(self.N) if i not in set(seeds)], dtype=np.int64)
        picks = self.rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
        return seeds + [int(p) for p in picks]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    static_graph: Dict[str, torch.Tensor],
    sample_event_feat: torch.Tensor,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load weights into a model instance. Runs a dry forward pass first so that
    lazy layers (e.g. GATv2Conv(in_channels=-1)) materialize their parameter
    shapes before `load_state_dict` is called.
    """
    model.to(device)
    N = static_graph["x_static"].shape[0]
    node_dim = static_graph["x_static"].shape[1] + 1  # +1 for the seed-mask column

    dummy_x = torch.zeros((1, N, node_dim), dtype=torch.float, device=device)
    dummy_event = sample_event_feat.to(device).unsqueeze(0)

    with torch.no_grad():
        _ = model(dummy_x, static_graph, dummy_event)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sweep evaluator
# ---------------------------------------------------------------------------


class SpeedEvaluator:
    """
    Runs a lam / lam_d sweep and records speed-of-infection scores per
    (selector, event).

    Seed selection is cached per (selector_name, event_id) since none of the
    selectors depend on lam / lam_d -- only the SCM dynamics do. A single
    SpeedSimulator is built per event and its beta tensors are swapped in
    place across sweep values to avoid rebuilding the sparse adjacency.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index_1: torch.Tensor,
        triangles_list: List,
        event_prob_columns: Dict[int, np.ndarray],
        event_feats: Dict[int, torch.Tensor],
        k_seeds: int = 2,
        num_mc_trials: int = 50,
        decay: float = 0.3,
        infected_target: int = 10,
        max_sim_steps: int = 50,
        device: str = "cpu",
    ):
        self.N = num_nodes
        self.edge_index_1 = edge_index_1
        self.triangles_list = triangles_list
        self.event_prob_columns = event_prob_columns
        self.event_feats = event_feats
        self.k_seeds = k_seeds
        self.num_mc_trials = num_mc_trials
        self.decay = decay
        self.infected_target = infected_target
        self.max_sim_steps = max_sim_steps
        self.device = device
        self._seed_cache: Dict[tuple, List[int]] = {}

    def _get_seeds(self, name: str, selector, event_id: int) -> List[int]:
        key = (name, int(event_id))
        if key not in self._seed_cache:
            feat = self.event_feats[int(event_id)]
            self._seed_cache[key] = selector.select(feat, k=self.k_seeds)
        return self._seed_cache[key]

    def run_sweep(
        self,
        selectors: Dict[str, object],
        axis: str,
        values: Iterable[float],
        fixed_value: float,
        event_ids: Sequence[int],
        verbose: bool = True,
    ) -> Dict[str, Dict[float, np.ndarray]]:
        assert axis in ("lam", "lam_d"), "axis must be 'lam' or 'lam_d'"
        values = [float(v) for v in values]
        results: Dict[str, Dict[float, List[float]]] = {
            name: {v: [] for v in values} for name in selectors
        }

        for name, sel in selectors.items():
            for eid in event_ids:
                self._get_seeds(name, sel, eid)

        for ei, eid in enumerate(event_ids):
            p = self.event_prob_columns[int(eid)]
            p_tensor = torch.tensor(p, dtype=torch.float, device=self.device)

            sim = SpeedSimulator(
                num_nodes=self.N,
                edge_index_1=self.edge_index_1,
                triangles_list=self.triangles_list,
                beta_tensor=p_tensor.clone(),
                beta_delta_tensor=p_tensor.clone(),
                num_mc_trials=self.num_mc_trials,
                device=self.device,
            )

            for v in values:
                lam = v if axis == "lam" else fixed_value
                lam_d = v if axis == "lam_d" else fixed_value
                sim.beta = lam * p_tensor
                sim.beta_delta = lam_d * p_tensor

                for name, sel in selectors.items():
                    seeds = self._get_seeds(name, sel, eid)
                    score = sim.speed_score(
                        initial_infected=seeds,
                        target=self.infected_target,
                        decay=self.decay,
                        max_steps=self.max_sim_steps,
                    )
                    results[name][v].append(score)

            if verbose:
                print(
                    f"  [{ei + 1}/{len(event_ids)}] event {int(eid)} done "
                    f"({axis} sweep, {len(values)} values)",
                    flush=True,
                )
            del sim

        return {
            name: {v: np.array(scores, dtype=np.float64) for v, scores in per_val.items()}
            for name, per_val in results.items()
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweeps(
    results_lam: Dict[str, Dict[float, np.ndarray]],
    results_lam_d: Dict[str, Dict[float, np.ndarray]],
    fixed_lam: float,
    fixed_lam_d: float,
    decay: float = 0.3,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    panels = [
        (axes[0], results_lam, r"$\lambda$", r"$\lambda_\Delta$", fixed_lam_d),
        (axes[1], results_lam_d, r"$\lambda_\Delta$", r"$\lambda$", fixed_lam),
    ]

    for ax, results, axis_label, other_label, other_val in panels:
        for name, per_val in results.items():
            xs = sorted(per_val.keys())
            means = np.array([per_val[x].mean() for x in xs])
            sems = np.array([
                per_val[x].std(ddof=1) / max(1.0, np.sqrt(len(per_val[x])))
                if len(per_val[x]) > 1 else 0.0
                for x in xs
            ])
            ax.plot(xs, means, marker="o", label=name, linewidth=1.8)
            ax.fill_between(xs, means - sems, means + sems, alpha=0.15)
        ax.set_xlabel(axis_label)
        ax.set_title(f"Speed vs {axis_label}   ({other_label} = {other_val})")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"Speed score  (decay = {decay})")
    axes[0].legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return fig
