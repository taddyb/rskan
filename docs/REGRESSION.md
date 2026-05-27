# Pinned regression tests

These tests are the rskan-side analogue of ddrs's `compare_ddr_sandbox`
"ABSOLUTE MATCH" gate. They MUST NEVER go red without a deliberate, documented
spec change. If you break one of these, fix it before merging — do not loosen
the tolerance.

| # | Test                                                                                   | What it guards                                                  |
| - | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1 | `cargo test -p rskan --test parity_forward -- ddr_scale_must_match_pykan`              | DDR-scale (21×21) forward parity vs pykan — the ddrs contract.  |
| 2 | `cargo test -p rskan --test parity_forward -- fixture_sweep_forward`                   | All bare-KANLayer forward cases (small/standard/OOD/degenerate). |
| 3 | `cargo test -p rskan --test parity_backward -- fixture_sweep_backward`                 | All bare-KANLayer backward cases (4 grads each).                |
| 4 | `cargo test -p rskan --test kan_stack -- fixture_sweep_kan_stack`                      | Multi-layer Kan + per-layer trajectory match.                   |
| 5 | `cargo test -p rskan --test init_smoke`                                                | Init reproducibility, shape correctness, require_grad flags.    |
| 6 | `cd ~/projects/ddr && uv run pytest ~/projects/rskan/rskan-py/tests/ -v`               | Python forward + backward parity vs pykan.                      |
| 7 | `cargo run -p rskan --example tiny_regression --release`                               | End-to-end training smoke — Adam converges to fit a smooth fn.  |

To regenerate fixtures (required if pykan version changes):

```bash
cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
```

Then re-run tests 1-6 above to confirm parity holds against the updated fixtures.
