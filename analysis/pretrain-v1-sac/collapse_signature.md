# World-collapse signature summary
§4.7 says collapse is the JOINT appearance of three effects. This table reports each condition's value for each effect; the headline comparison is collapse_prone vs the two baselines (frozen_wm, balanced_replay).

| Condition | n seeds | entropy slope (Δ/iter) | gap_h5 (final) | E_visited slope | E_underv slope | trained success (final) | holdout success (final) | shift drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| collapse_prone | 3 | -0.0024 ± 0.0003 | +0.019 ± 0.028 | +0.0001 | +0.0001 | 0.000 | 0.000 | +0.000 |
| balanced_replay | 3 | -0.0009 ± 0.0005 | +0.000 ± 0.000 | -0.0001 | +nan | 0.000 | 0.000 | +0.000 |
| frozen_wm | 3 | -0.0020 ± 0.0001 | +0.029 ± 0.006 | -0.0000 | -0.0001 | 0.000 | 0.000 | +0.000 |

### Reading the signature

- **entropy slope** < 0 → visitation narrowing (proposal §4.5.1).
- **gap_h5** > 0 and growing → world model is selectively worse off-support at horizon 5 (proposal §4.5.2).
- **E_visited slope** ≈ 0 while **E_underv slope** > 0 → forgetting is localized, not uniform.
- **shift drop** > 0 → behavioral consequence of goal shift (proposal §4.5.3).

Collapse claim is supported iff collapse_prone shows all four AND the baselines do not.
