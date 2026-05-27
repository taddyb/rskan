# rskan parity fixtures

These `.npy` files are byte-for-byte exports of pykan's KAN module outputs and
gradients, used as the ground truth for rskan's numerical parity tests.

**Regenerate:**

```bash
cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
```

Requires pykan installed in DDR's uv venv (DDR pins the version in
`~/projects/ddr/uv.lock`).

## Layout

```
fixtures/
├── manifest.json                  # machine-readable list of all cases
├── README.md                      # this file
├── kanlayer_*/                    # bare KANLayer cases
│   ├── params.json
│   ├── grid.npy   coef.npy   scale_base.npy   scale_sp.npy   mask.npy
│   ├── x.npy   y.npy
│   └── grad_x.npy   grad_coef.npy   grad_scale_base.npy   grad_scale_sp.npy
└── kan_*/                         # multi-layer cases
    ├── params.json
    ├── x.npy   y.npy
    ├── trajectory_l<N>.npy        # per-layer outputs (one file per layer)
    └── layer_<N>/{grid,coef,scale_base,scale_sp,mask,grad_coef,...}.npy
```

Never edit these files by hand. CI never invokes Python.
