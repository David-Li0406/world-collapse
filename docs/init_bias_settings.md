# World-collapse settings: init-bias amplification (no permanent split)

## Motivation
The existing Phase-B conditions (`collapse_prone`, `balanced_replay`, `frozen_wm`)
keep a **permanent** trained/holdout goal split for the whole run. That is a clean
*instrument* but unrealistic: real systems are not permanently forbidden a region.
These two new settings instead **seed a small bias only at the beginning** and let
the policy↔WM feedback loop **amplify** it on the *full* goal space, then measure
whether that endogenous amplification produces collapse.

The goal space is split (for measurement only) at goal-x = `split` into
region **A** (lower half) and region **B** (upper half). The split is used to
*seed* the bias and to *measure* it — **not** to restrict training after the seed.

## Setting 1 — policy-seeded bias (`bias_init_policy`)
- **Seed:** during the **first feedback iteration only** (`frame < bias_warmup_frames`,
  default 60k = one K=5 macro-iter), training goals are restricted to region A.
  After that, training goals are sampled from the **full** space.
- **Idea:** the policy starts skewed toward A (it only practiced A in iter 1);
  observe whether the loop enlarges that skew rather than letting the now-unbiased
  goal stream re-broaden it.
- Mounted models: standard unbiased round-0 `(M0, Policy0)`.

## Setting 2 — WM-seeded bias (`bias_init_wm`)
- **Seed:** **Phase 0 (round-0)** collects an **A-skewed** real buffer
  (`collect_bias_prob` of episodes draw the goal from A; default 0.8), so the
  resulting world model **M0 is more accurate in A than B** from the start.
- **Phase B:** trains on the **full** goal space throughout (no warmup, no split).
- **Idea:** the WM starts skewed; observe whether the loop enlarges that skew.

### ⚠ Open design fork (Setting 2 isolation)
A biased round-0 trains **both** M0 *and* Policy0 on the A-skewed buffer, so by
default **both** inherit the bias — this is "phase-0-seeded" bias, not cleanly
"WM-only". To isolate a *WM-only* seed, Phase B must mount the biased M0 but pair
it with an **unbiased** policy (from a separate clean round-0). Decision pending.

## Decoupling training bias from measurement (shared infra)
- New flag `measure_split: true` defines fixed measurement regions A/B and runs
  `goal_shift_eval` every eval **regardless of the training goal distribution**.
- `measure_bias_fraction` (default 0.5) sets the A/B split-x; written to
  `cfg.static_goal_split` so the coverage probe partition (static_visited = goal-x<split)
  aligns with A.
- Training goal distribution is controlled separately, per episode:
  - round-0: oversample A with prob `collect_bias_prob` (else full space).
  - online + `bias_goal: true`  → permanent A (legacy conditions).
  - online + `bias_warmup_frames>0` → A while `frame < warmup`, else full (Setting 1).
  - else → full space (Setting 2 Phase B).

## WM regime is orthogonal (compose via `--condition`)
Each bias setting is run under a WM-update regime:
- `collapse_prone` (recency-window WM) — the **amplifier** (expected to collapse).
- `frozen_wm` (WM never updates) — the **control** (cannot forget → should NOT
  amplify the seed). The contrast frozen-vs-recency is the causal test that the
  loop (via WM forgetting) is what enlarges the bias.

## Metrics that demonstrate the collapse (tracked over the K iterations)
1. **Bias amplification (headline):** `beh/shift_success_drop = SR_A − SR_B`.
   Starts ~0 (small seed) → grows over iters = amplification. Plus `beh/trained_success_rate`
   (A) and `beh/holdout_success_rate` (B) individually.
2. **Coverage narrowing:** `cov/visitation_entropy` ↓ and new
   `cov/frac_goal_in_A` / `cov/frac_obj_in_A` ↑ — the policy increasingly *generates*
   A-skewed data even though goals are requested full-space.
3. **WM forgetting of B:** `wm/static_gap_h{1,5,10}` (errB − errA) ↑ and
   `wm/forget_h*_static_underv` (forgetting in B vs M0) ↑.
The collapse signature = all three rise together from a small seed under
`collapse_prone`, and stay ~flat under `frozen_wm` (control).

## New config / workflow surface
- `configs/bias_init_policy_drqv2.yaml` (Setting 1 overlay; self-contained K=5/300k).
- `configs/bias_init_wm_drqv2.yaml` (Setting 2 Phase-B overlay).
- `configs/round0_wmbias_drqv2.yaml` (A-skewed round-0 for Setting 2).
- `setup-shared-drqv2.yml`: new `round0_overlay` input (to build the biased setup).
- `run-conditions-drqv2.yml`: reuse `extra_overlay` (= `bias_init_policy` / `bias_init_wm`).

## Code touch-points
- `src/wcollapse/training/online_drqv2.py`: measurement-split block (decoupled),
  per-episode training-goal controller (round0 collect-bias + online warmup),
  eval gating on measurement regions.
- `src/wcollapse/eval/coverage_drqv2.py`: add `frac_goal_in_A` / `frac_obj_in_A`.
