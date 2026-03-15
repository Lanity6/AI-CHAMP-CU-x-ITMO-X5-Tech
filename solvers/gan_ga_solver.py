"""
GAN-based Genetic Algorithm 3D Bin Packing Solver — X5 Tech Smart Packing

Based on: "A novel approach for solving 3D bin packing problem by integrating
a generative adversarial network with a genetic algorithm"
(Scientific Reports, 2024, doi:10.1038/s41598-024-56699-7)

Adaptation for single-pallet packing:
  - Permutation encoding: chromosome = item ordering + rotation choices
  - Greedy decoder: extreme-points placement following chromosome order
  - GA: tournament selection, order crossover (OX), swap/insert/rotation mutation
  - GAN component: lightweight numpy-based generator/discriminator for diversity
    injection and fitness augmentation (no pytorch dependency)

Scoring: maximizes final_score = 0.50*vol + 0.30*coverage + 0.10*fragility + 0.10*time
"""

import json
import sys
import time
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from copy import deepcopy

import numpy as np

from solvers.greedy_algorithm import (
    get_rotations,
    PlacedGrid,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    "time_budget_s":        0.85,     # under 1s → time_score=1.0
    "population_size":      14,       # GA population size
    "tournament_size":      3,        # tournament selection size
    "crossover_rate":       0.85,     # probability of crossover
    "mutation_rate":        0.30,     # probability of mutation (higher for more exploration)
    "elitism_count":        2,        # number of elite individuals preserved
    "gan_interval":         3,        # inject GAN solutions every N generations
    "gan_inject_count":     2,        # how many GAN solutions to inject
    "gan_hidden_size":      16,       # GAN hidden layer size
    "gan_noise_dim":        8,        # GAN noise vector dimension
    "gan_train_epochs":     2,        # GAN training epochs per interval
    "gan_lr":               0.01,     # GAN learning rate
    "alpha_discriminator":  0.03,     # weight for discriminator fitness augmentation
    "multi_start_count":    15,       # initial random restarts for seeding
    "eps_limit":            50,       # max extreme points in GA decode (speed)
    "eps_limit_final":      200,      # max extreme points in final decode (quality)
}

# ──────────────────────────────────────────────────────────────────────────────
# Item expansion and sorting helpers
# ──────────────────────────────────────────────────────────────────────────────

def expand_items(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand all SKU quantities into individual item dicts."""
    items = []
    for box in request["boxes"]:
        for idx in range(box["quantity"]):
            items.append({
                "sku_id": box["sku_id"],
                "instance_index": idx,
                "l": box["length_mm"],
                "w": box["width_mm"],
                "h": box["height_mm"],
                "weight": box["weight_kg"],
                "volume": box["length_mm"] * box["width_mm"] * box["height_mm"],
                "strict_upright": box["strict_upright"],
                "fragile": box["fragile"],
                "stackable": box.get("stackable", True),
            })
    return items


def default_sort_key(item: Dict) -> Tuple:
    """3-level sort: normal -> fragile -> non-stackable, then by volume desc."""
    group = 0
    if item["fragile"]:
        group = 1
    if not item["stackable"]:
        group = 2
    return (group, -item["volume"])


# ──────────────────────────────────────────────────────────────────────────────
# Greedy decoder: given item order + rotation choices, place items on pallet
# ──────────────────────────────────────────────────────────────────────────────

def decode_chromosome(
    items: List[Dict[str, Any]],
    perm: List[int],
    rot_choices: List[int],
    PL: int, PW: int, PH: int,
    max_weight: float,
    eps_limit: int = 80,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float, int]:
    """
    Decode a chromosome (permutation + rotation choices) into a packing solution.
    Uses limited extreme points for speed.

    Returns: (placements, unplaced, packed_volume, total_weight, frag_violations)
    """
    grid = PlacedGrid(PL, PW)
    placements: List[Dict[str, Any]] = []
    unplaced_map: Dict[str, Dict[str, Any]] = {}
    total_weight = 0.0
    packed_volume = 0.0
    frag_violations = 0

    eps: List[Tuple[int, int, int]] = [(0, 0, 0)]
    eps_set: set = {(0, 0, 0)}

    for order_idx, item_idx in enumerate(perm):
        item = items[item_idx]

        if total_weight + item["weight"] > max_weight + 1e-6:
            rec = unplaced_map.setdefault(item["sku_id"],
                                          {"count": 0, "reason": "weight_limit_exceeded"})
            rec["count"] += 1
            continue

        rotations = get_rotations(item["l"], item["w"], item["h"], item["strict_upright"])
        if not rotations:
            rec = unplaced_map.setdefault(item["sku_id"],
                                          {"count": 0, "reason": "no_space"})
            rec["count"] += 1
            continue

        # Use chromosome's rotation choice (modulo available rotations)
        rot_idx = rot_choices[order_idx] % len(rotations)
        # Try preferred rotation first, then others
        rot_order = [rot_idx] + [i for i in range(len(rotations)) if i != rot_idx]

        best_score: Optional[Tuple] = None
        best_placement: Optional[Tuple] = None

        # Limit extreme points to keep decode fast
        active_eps = eps if len(eps) <= eps_limit else eps[:eps_limit]

        for ri in rot_order:
            dl, dw, dh, code = rotations[ri]
            for (ex, ey, _ez) in active_eps:
                x2, y2 = ex + dl, ey + dw
                if x2 > PL or y2 > PW:
                    continue

                cands = grid.candidates_xy(ex, ey, x2, y2)
                z = grid.find_z(ex, ey, x2, y2, cands)
                if z is None:
                    continue
                if z + dh > PH:
                    continue
                if grid.collision_3d(ex, ey, z, x2, y2, z + dh, cands):
                    continue
                if not grid.check_support(ex, ey, z, x2, y2, dl * dw, cands):
                    continue
                if not grid.check_fragility(ex, ey, z, x2, y2, item["weight"], cands):
                    continue

                score = (z, ex, ey)
                if best_score is None or score < best_score:
                    best_score = score
                    best_placement = (ex, ey, z, dl, dw, dh, code)

            # Early exit if found placement at z=0 (floor level = best)
            if best_score is not None and best_score[0] == 0:
                break

        if best_placement is not None:
            px, py, pz, dl, dw, dh, code = best_placement

            box_record = {
                "sku_id": item["sku_id"],
                "x_min": px, "x_max": px + dl,
                "y_min": py, "y_max": py + dw,
                "z_min": pz, "z_max": pz + dh,
                "weight": item["weight"],
                "fragile": item["fragile"],
                "stackable": item["stackable"],
            }
            grid.add(box_record)
            total_weight += item["weight"]
            packed_volume += dl * dw * dh

            for ep in ((px + dl, py, pz), (px, py + dw, pz), (px, py, pz + dh)):
                if ep not in eps_set:
                    eps_set.add(ep)
                    eps.append(ep)

            # Check fragility violations for scoring
            if item["weight"] > 2.0 and pz > 0:
                x2p, y2p = px + dl, py + dw
                for b in grid.boxes[:-1]:
                    if b["fragile"] and b["z_max"] == pz:
                        if (x2p > b["x_min"] and b["x_max"] > px and
                                y2p > b["y_min"] and b["y_max"] > py):
                            frag_violations += 1

            placements.append({
                "sku_id": item["sku_id"],
                "instance_index": item["instance_index"],
                "position": {"x_mm": px, "y_mm": py, "z_mm": pz},
                "dimensions_placed": {"length_mm": dl, "width_mm": dw, "height_mm": dh},
                "rotation_code": code,
                "layer": 0,
            })
        else:
            rec = unplaced_map.setdefault(item["sku_id"],
                                          {"count": 0, "reason": "no_space"})
            rec["count"] += 1

    unplaced = [
        {"sku_id": sku_id, "quantity_unplaced": rec["count"], "reason": rec["reason"]}
        for sku_id, rec in unplaced_map.items()
    ]
    return placements, unplaced, packed_volume, total_weight, frag_violations


# ──────────────────────────────────────────────────────────────────────────────
# Fitness evaluation
# ──────────────────────────────────────────────────────────────────────────────

def compute_fitness(
    placements: List[Dict],
    packed_volume: float,
    total_items: int,
    frag_violations: int,
    pallet_vol: float,
    elapsed_ms: int,
) -> float:
    """Compute the competition score for a packing solution."""
    vol_util = packed_volume / pallet_vol if pallet_vol > 0 else 0.0
    item_coverage = len(placements) / total_items if total_items > 0 else 0.0
    fragility_score = max(0.0, 1.0 - 0.05 * frag_violations)
    if elapsed_ms <= 1000:
        time_score = 1.0
    elif elapsed_ms <= 5000:
        time_score = 0.7
    elif elapsed_ms <= 30000:
        time_score = 0.3
    else:
        time_score = 0.0

    return 0.50 * vol_util + 0.30 * item_coverage + 0.10 * fragility_score + 0.10 * time_score


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight GAN (numpy-based) — Generator + Discriminator
# Per the paper: GAN generates synthetic solutions; Discriminator augments fitness
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


class SimpleGAN:
    """
    Lightweight GAN with numpy. Generator produces feature vectors representing
    item-position preferences; Discriminator distinguishes real (elite) from
    synthetic solutions.

    Encoding: each solution is represented as a normalized position vector
    (item index / n_items for each position in the permutation).
    """

    def __init__(self, solution_dim: int, noise_dim: int = 16, hidden: int = 32,
                 lr: float = 0.01):
        self.solution_dim = solution_dim
        self.noise_dim = noise_dim
        self.hidden = hidden
        self.lr = lr

        # Generator: noise_dim -> hidden -> solution_dim
        scale_g = np.sqrt(2.0 / noise_dim)
        self.g_w1 = np.random.randn(noise_dim, hidden).astype(np.float32) * scale_g
        self.g_b1 = np.zeros(hidden, dtype=np.float32)
        self.g_w2 = np.random.randn(hidden, solution_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.g_b2 = np.zeros(solution_dim, dtype=np.float32)

        # Discriminator: solution_dim -> hidden -> 1
        scale_d = np.sqrt(2.0 / solution_dim)
        self.d_w1 = np.random.randn(solution_dim, hidden).astype(np.float32) * scale_d
        self.d_b1 = np.zeros(hidden, dtype=np.float32)
        self.d_w2 = np.random.randn(hidden, 1).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.d_b2 = np.zeros(1, dtype=np.float32)

    def _encode_perm(self, perm: List[int]) -> np.ndarray:
        """Encode permutation as normalized position vector."""
        n = len(perm)
        if n == 0:
            return np.zeros(self.solution_dim, dtype=np.float32)
        vec = np.zeros(self.solution_dim, dtype=np.float32)
        for pos, item_idx in enumerate(perm):
            if item_idx < self.solution_dim:
                vec[item_idx] = pos / max(n - 1, 1)
        return vec

    def _generator_forward(self, z: np.ndarray) -> np.ndarray:
        """Generator forward pass. Returns solution-dim vector in [0,1]."""
        h = _relu(z @ self.g_w1 + self.g_b1)
        out = _sigmoid(h @ self.g_w2 + self.g_b2)
        return out

    def _discriminator_forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Discriminator forward pass. Returns (output, hidden)."""
        h = _leaky_relu(x @ self.d_w1 + self.d_b1)
        out = _sigmoid(h @ self.d_w2 + self.d_b2)
        return out, h

    def discriminator_score(self, perm: List[int]) -> float:
        """D(x) score for a permutation — how 'real' (elite-like) it looks."""
        x = self._encode_perm(perm).reshape(1, -1)
        out, _ = self._discriminator_forward(x)
        return float(out[0, 0])

    def generate_perm(self, n_items: int) -> List[int]:
        """Generate a permutation from the generator network."""
        z = np.random.randn(1, self.noise_dim).astype(np.float32)
        raw = self._generator_forward(z)[0]

        # Convert continuous output to permutation via argsort
        # Only use first n_items dimensions
        scores = raw[:n_items]
        perm = list(np.argsort(-scores))  # descending sort = priority
        return perm

    def train_step(self, real_perms: List[List[int]], batch_size: int = 8) -> None:
        """One training step: update discriminator then generator."""
        if not real_perms:
            return

        # Encode real solutions
        real_batch = []
        for _ in range(min(batch_size, len(real_perms))):
            p = random.choice(real_perms)
            real_batch.append(self._encode_perm(p))
        real_x = np.array(real_batch, dtype=np.float32)

        # Generate fake solutions
        z = np.random.randn(len(real_batch), self.noise_dim).astype(np.float32)
        fake_x = self._generator_forward(z)

        # ── Train Discriminator ──
        # Real labels = 1, Fake labels = 0
        d_real, h_real = self._discriminator_forward(real_x)
        d_fake, h_fake = self._discriminator_forward(fake_x)

        # Binary cross-entropy gradients for discriminator
        eps = 1e-7
        # dL/d_out for real: -1/d_real, for fake: 1/(1-d_fake)
        grad_d_real = -(1.0 / (d_real + eps))
        grad_d_fake = 1.0 / (1.0 - d_fake + eps)

        # Backprop through sigmoid: d_sigmoid/dx = out*(1-out)
        grad_pre_real = grad_d_real * d_real * (1 - d_real)
        grad_pre_fake = grad_d_fake * d_fake * (1 - d_fake)

        # Update d_w2, d_b2
        grad_dw2 = h_real.T @ grad_pre_real / len(real_batch)
        grad_dw2 += h_fake.T @ grad_pre_fake / len(real_batch)
        grad_db2 = (grad_pre_real.mean(axis=0) + grad_pre_fake.mean(axis=0))

        self.d_w2 -= self.lr * grad_dw2
        self.d_b2 -= self.lr * grad_db2.flatten()

        # Backprop through leaky_relu to d_w1
        grad_h_real = grad_pre_real @ self.d_w2.T
        grad_h_fake = grad_pre_fake @ self.d_w2.T
        grad_h_real *= np.where(h_real > 0, 1, 0.01)
        grad_h_fake *= np.where(h_fake > 0, 1, 0.01)

        grad_dw1 = real_x.T @ grad_h_real / len(real_batch)
        grad_dw1 += fake_x.T @ grad_h_fake / len(real_batch)
        grad_db1 = grad_h_real.mean(axis=0) + grad_h_fake.mean(axis=0)

        self.d_w1 -= self.lr * grad_dw1
        self.d_b1 -= self.lr * grad_db1

        # ── Train Generator ──
        # Want D(G(z)) -> 1, so loss = -log(D(G(z)))
        z2 = np.random.randn(batch_size, self.noise_dim).astype(np.float32)
        fake_x2 = self._generator_forward(z2)
        d_gen, h_gen = self._discriminator_forward(fake_x2)

        # dL/d_out = -1/d_gen
        grad_d_gen = -(1.0 / (d_gen + eps))
        grad_pre_gen = grad_d_gen * d_gen * (1 - d_gen)

        # Through discriminator (frozen conceptually, but we just use gradient flow)
        grad_h_gen = grad_pre_gen @ self.d_w2.T
        grad_h_gen *= np.where(h_gen > 0, 1, 0.01)
        grad_fake_x = grad_h_gen @ self.d_w1.T

        # Through generator sigmoid: dsigmoid/dx = out*(1-out)
        grad_pre_g2 = grad_fake_x * fake_x2 * (1 - fake_x2)
        h_g = _relu(z2 @ self.g_w1 + self.g_b1)
        grad_gw2 = h_g.T @ grad_pre_g2 / batch_size
        grad_gb2 = grad_pre_g2.mean(axis=0)

        self.g_w2 -= self.lr * grad_gw2
        self.g_b2 -= self.lr * grad_gb2

        # Through relu
        grad_h_g = grad_pre_g2 @ self.g_w2.T
        grad_h_g *= np.where(h_g > 0, 1, 0)
        grad_gw1 = z2.T @ grad_h_g / batch_size
        grad_gb1 = grad_h_g.mean(axis=0)

        self.g_w1 -= self.lr * grad_gw1
        self.g_b1 -= self.lr * grad_gb1


# ──────────────────────────────────────────────────────────────────────────────
# GA Operations
# ──────────────────────────────────────────────────────────────────────────────

def tournament_select(
    population: List[Tuple[List[int], List[int]]],
    fitnesses: List[float],
    k: int,
) -> Tuple[List[int], List[int]]:
    """Tournament selection: pick k random, return the fittest."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    perm, rots = population[best_idx]
    return list(perm), list(rots)


def order_crossover(
    p1: List[int], p2: List[int]
) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX) for permutation encoding."""
    n = len(p1)
    if n <= 2:
        return list(p1), list(p2)

    cx1, cx2 = sorted(random.sample(range(n), 2))

    # Child 1: segment from p1, rest from p2
    child1 = [-1] * n
    child1[cx1:cx2] = p1[cx1:cx2]
    used1 = set(child1[cx1:cx2])
    fill = [g for g in p2 if g not in used1]
    j = 0
    for i in range(n):
        if child1[i] == -1:
            child1[i] = fill[j]
            j += 1

    # Child 2: segment from p2, rest from p1
    child2 = [-1] * n
    child2[cx1:cx2] = p2[cx1:cx2]
    used2 = set(child2[cx1:cx2])
    fill = [g for g in p1 if g not in used2]
    j = 0
    for i in range(n):
        if child2[i] == -1:
            child2[i] = fill[j]
            j += 1

    return child1, child2


def rotation_crossover(
    r1: List[int], r2: List[int]
) -> Tuple[List[int], List[int]]:
    """Two-point crossover for rotation choices (as in the paper)."""
    n = len(r1)
    if n <= 2:
        return list(r1), list(r2)
    cx1, cx2 = sorted(random.sample(range(n), 2))
    c1 = r1[:cx1] + r2[cx1:cx2] + r1[cx2:]
    c2 = r2[:cx1] + r1[cx1:cx2] + r2[cx2:]
    return c1, c2


def mutate_perm(perm: List[int], rate: float) -> List[int]:
    """Apply swap, insert, or reverse mutation to permutation."""
    if random.random() > rate or len(perm) <= 2:
        return perm
    perm = list(perm)
    op = random.randint(0, 2)
    if op == 0:  # swap
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    elif op == 1:  # insert
        i = random.randrange(len(perm))
        j = random.randrange(len(perm))
        item = perm.pop(i)
        perm.insert(j, item)
    else:  # reverse segment
        i, j = sorted(random.sample(range(len(perm)), 2))
        perm[i:j+1] = perm[i:j+1][::-1]
    return perm


def mutate_rotations(rots: List[int], rate: float, max_rot: int = 6) -> List[int]:
    """Randomly flip some rotation choices."""
    if random.random() > rate:
        return rots
    rots = list(rots)
    n_flips = max(1, len(rots) // 10)
    for _ in range(n_flips):
        i = random.randrange(len(rots))
        rots[i] = random.randrange(max_rot)
    return rots


# ──────────────────────────────────────────────────────────────────────────────
# Multi-start initialization
# ──────────────────────────────────────────────────────────────────────────────

def create_initial_population(
    items: List[Dict], n_items: int, pop_size: int, multi_start: int,
    PL: int, PW: int, PH: int, max_weight: float, pallet_vol: float,
    total_items: int, time_start: float, time_budget: float,
) -> List[Tuple[List[int], List[int], float]]:
    """Generate initial population via random restarts, keep best pop_size."""
    candidates = []

    # Default sorted order (greedy baseline)
    default_perm = list(range(n_items))
    default_items_sorted = sorted(range(n_items), key=lambda i: default_sort_key(items[i]))
    default_rots = [0] * n_items

    # Pre-compute several sort orders for diversity
    sort_by_vol = sorted(range(n_items), key=lambda i: -items[i]["volume"])
    sort_by_weight = sorted(range(n_items), key=lambda i: -items[i]["weight"])
    sort_by_height = sorted(range(n_items), key=lambda i: -items[i]["h"])
    sort_by_area = sorted(range(n_items), key=lambda i: -(items[i]["l"] * items[i]["w"]))

    seed_perms = [
        default_items_sorted,
        sort_by_vol,
        sort_by_weight,
        sort_by_height,
        sort_by_area,
    ]

    for trial in range(multi_start):
        if time.time() - time_start > time_budget * 0.20:
            break

        if trial < len(seed_perms):
            perm = list(seed_perms[trial])
            rots = list(default_rots)
        else:
            # Start from a random seed ordering and partially perturb
            base = list(random.choice(seed_perms))
            n_swaps = random.randint(1, max(1, n_items // 4))
            for _ in range(n_swaps):
                i, j = random.sample(range(n_items), 2)
                base[i], base[j] = base[j], base[i]
            perm = base
            rots = [random.randrange(6) for _ in range(n_items)]

        plcs, _, pvol, _, fviol = decode_chromosome(
            items, perm, rots, PL, PW, PH, max_weight, eps_limit=60
        )
        elapsed_ms = int((time.time() - time_start) * 1000)
        fit = compute_fitness(plcs, pvol, total_items, fviol, pallet_vol, elapsed_ms)
        candidates.append((perm, rots, fit))

    # Keep top pop_size
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[:pop_size]


# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_task(request: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.time()

    pallet = request["pallet"]
    PL = pallet["length_mm"]
    PW = pallet["width_mm"]
    PH = pallet["max_height_mm"]
    max_weight = pallet["max_weight_kg"]
    pallet_vol = PL * PW * PH

    items = expand_items(request)
    n_items = len(items)
    total_items = n_items

    # Adaptive parameters based on problem size
    time_budget = CONFIG["time_budget_s"]
    eps_fast = CONFIG["eps_limit"]
    if n_items > 120:
        # Large instances: reduce budget, use fewer eps to stay under 1s
        time_budget = min(time_budget, 0.70)
        eps_fast = 35
    elif n_items > 80:
        time_budget = min(time_budget, 0.80)
        eps_fast = 45

    pop_size = CONFIG["population_size"]
    tournament_k = CONFIG["tournament_size"]
    cx_rate = CONFIG["crossover_rate"]
    mut_rate = CONFIG["mutation_rate"]
    elite_count = CONFIG["elitism_count"]
    gan_interval = CONFIG["gan_interval"]
    gan_inject = CONFIG["gan_inject_count"]
    alpha_d = CONFIG["alpha_discriminator"]

    if n_items == 0:
        return {
            "task_id": request["task_id"],
            "solver_version": "gan-ga-1.0",
            "solve_time_ms": 0,
            "placements": [],
            "unplaced": [],
        }

    # ── Phase 0: Quick greedy baseline (always fast) ──
    baseline_perm = sorted(range(n_items), key=lambda i: default_sort_key(items[i]))
    baseline_rots = [0] * n_items
    best_placements, best_unplaced, best_pvol, _, best_fviol = decode_chromosome(
        items, baseline_perm, baseline_rots, PL, PW, PH, max_weight, eps_limit=200
    )
    best_fitness = compute_fitness(
        best_placements, best_pvol, total_items, best_fviol, pallet_vol,
        int((time.time() - t_start) * 1000)
    )
    best_perm = list(baseline_perm)
    best_rots = list(baseline_rots)

    # ── Phase 1: Multi-start initialization for GA population ──
    pop_with_fit = create_initial_population(
        items, n_items, pop_size, CONFIG["multi_start_count"],
        PL, PW, PH, max_weight, pallet_vol, total_items, t_start, time_budget,
    )

    # Check if multi-start found better
    for p, r, f in pop_with_fit:
        if f > best_fitness:
            best_fitness = f
            best_perm = list(p)
            best_rots = list(r)

    # Separate into population and fitnesses
    population: List[Tuple[List[int], List[int]]] = [(p, r) for p, r, _ in pop_with_fit]
    fitnesses: List[float] = [f for _, _, f in pop_with_fit]

    # Always include baseline in population
    population.append((list(baseline_perm), list(baseline_rots)))
    fitnesses.append(best_fitness)

    # Pad if needed
    while len(population) < pop_size:
        perm = list(baseline_perm)
        n_swaps = random.randint(1, max(1, n_items // 5))
        for _ in range(n_swaps):
            i, j = random.sample(range(n_items), 2)
            perm[i], perm[j] = perm[j], perm[i]
        rots = [random.randrange(6) for _ in range(n_items)]
        population.append((perm, rots))
        fitnesses.append(0.0)

    # ── Initialize GAN ──
    gan = SimpleGAN(
        solution_dim=n_items,
        noise_dim=CONFIG["gan_noise_dim"],
        hidden=CONFIG["gan_hidden_size"],
        lr=CONFIG["gan_lr"],
    )

    generation = 0

    # ── Phase 2: GA loop with GAN diversity injection ──
    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_budget * 0.80:
            break

        generation += 1

        # ── GAN training + injection (per paper: every k generations) ──
        if generation % gan_interval == 0 and elapsed < time_budget * 0.6:
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            elite_perms = [population[i][0] for i in sorted_indices[:max(3, elite_count)]]

            for _ in range(CONFIG["gan_train_epochs"]):
                gan.train_step(elite_perms, batch_size=min(6, len(elite_perms)))

            gan_solutions = []
            for _ in range(gan_inject):
                if time.time() - t_start >= time_budget * 0.7:
                    break
                gen_perm = gan.generate_perm(n_items)
                gen_rots = [random.randrange(6) for _ in range(n_items)]
                plcs, _, pvol, _, fviol = decode_chromosome(
                    items, gen_perm, gen_rots, PL, PW, PH, max_weight,
                    eps_limit=eps_fast,
                )
                elapsed_ms = int((time.time() - t_start) * 1000)
                fit = compute_fitness(plcs, pvol, total_items, fviol, pallet_vol, elapsed_ms)
                gan_solutions.append((gen_perm, gen_rots, fit))

                if fit > best_fitness:
                    best_fitness = fit
                    best_perm = list(gen_perm)
                    best_rots = list(gen_rots)

            worst_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:len(gan_solutions)]
            for wi, (gp, gr, gf) in zip(worst_indices, gan_solutions):
                population[wi] = (gp, gr)
                fitnesses[wi] = gf

        # ── Selection + Crossover + Mutation ──
        new_pop: List[Tuple[List[int], List[int]]] = []
        new_fit: List[float] = []

        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        for ei in sorted_indices[:elite_count]:
            new_pop.append(population[ei])
            new_fit.append(fitnesses[ei])

        while len(new_pop) < pop_size:
            if time.time() - t_start >= time_budget * 0.80:
                break

            parent1_perm, parent1_rots = tournament_select(population, fitnesses, tournament_k)
            parent2_perm, parent2_rots = tournament_select(population, fitnesses, tournament_k)

            if random.random() < cx_rate:
                child1_perm, child2_perm = order_crossover(parent1_perm, parent2_perm)
                child1_rots, child2_rots = rotation_crossover(parent1_rots, parent2_rots)
            else:
                child1_perm, child2_perm = list(parent1_perm), list(parent2_perm)
                child1_rots, child2_rots = list(parent1_rots), list(parent2_rots)

            child1_perm = mutate_perm(child1_perm, mut_rate)
            child1_rots = mutate_rotations(child1_rots, mut_rate)
            child2_perm = mutate_perm(child2_perm, mut_rate)
            child2_rots = mutate_rotations(child2_rots, mut_rate)

            for cperm, crots in [(child1_perm, child1_rots), (child2_perm, child2_rots)]:
                if len(new_pop) >= pop_size:
                    break
                if time.time() - t_start >= time_budget * 0.80:
                    break

                plcs, _, pvol, _, fviol = decode_chromosome(
                    items, cperm, crots, PL, PW, PH, max_weight,
                    eps_limit=eps_fast,
                )
                elapsed_ms = int((time.time() - t_start) * 1000)
                fit = compute_fitness(plcs, pvol, total_items, fviol, pallet_vol, elapsed_ms)

                # Fitness augmentation per paper eq. 6: F' = F + α*D(x)
                d_score = gan.discriminator_score(cperm)
                augmented_fit = fit + alpha_d * d_score

                new_pop.append((cperm, crots))
                new_fit.append(augmented_fit)

                if fit > best_fitness:
                    best_fitness = fit
                    best_perm = list(cperm)
                    best_rots = list(crots)

        population = new_pop
        fitnesses = new_fit

    # ── Final decode with higher quality (more extreme points) ──
    placements, unplaced, packed_volume, total_weight, frag_violations = decode_chromosome(
        items, best_perm, best_rots, PL, PW, PH, max_weight,
        eps_limit=CONFIG["eps_limit_final"]
    )

    # If final decode is worse than baseline (rare), use baseline
    final_fit = compute_fitness(
        placements, packed_volume, total_items, frag_violations, pallet_vol,
        int((time.time() - t_start) * 1000)
    )
    if final_fit < best_fitness * 0.98:
        # Re-decode baseline for final output
        placements, unplaced, _, _, _ = decode_chromosome(
            items, baseline_perm, baseline_rots, PL, PW, PH, max_weight,
            eps_limit=CONFIG["eps_limit_final"]
        )

    solve_time_ms = int((time.time() - t_start) * 1000)

    return {
        "task_id": request["task_id"],
        "solver_version": "gan-ga-1.0",
        "solve_time_ms": solve_time_ms,
        "placements": placements,
        "unplaced": unplaced,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    # Parse optional flags
    hq = "--hq" in sys.argv
    if hq:
        CONFIG["time_budget_s"] = 10.0
        CONFIG["population_size"] = 60
        CONFIG["multi_start_count"] = 120
        CONFIG["gan_train_epochs"] = 12
        CONFIG["gan_inject_count"] = 10

    from validator import evaluate_solution

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        tasks = json.load(f)

    resps = []
    ts = 0.0
    vc = 0
    for i, task in enumerate(tasks):
        resp = solve_task(task)
        resps.append(resp)
        r = evaluate_solution(task, resp)
        tid = task["task_id"]
        if r["valid"]:
            s = r["final_score"]
            ts += s
            vc += 1
            m = r["metrics"]
            resp["metrics"] = {
                "valid": True, "final_score": s,
                "volume_utilization": m["volume_utilization"],
                "item_coverage": m["item_coverage"],
                "fragility_score": m["fragility_score"],
                "time_score": m["time_score"],
            }
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: score={s:.4f} "
                  f"vol={m['volume_utilization']:.4f} cov={m['item_coverage']:.4f} "
                  f"frag={m['fragility_score']:.4f} time={m['time_score']:.4f} "
                  f"({len(resp['placements'])} placed, {resp['solve_time_ms']}ms)")
        else:
            resp["metrics"] = {"valid": False, "final_score": 0.0, "error": r["error"]}
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: INVALID - {r['error']}")

    avg = ts / len(tasks) if tasks else 0.0
    print(f"\n{'='*60}\nTasks: {len(tasks)} | Valid: {vc} | Avg score: {avg:.4f}")
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(resps, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {sys.argv[2]}")


if __name__ == "__main__":
    main()
