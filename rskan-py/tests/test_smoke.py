"""Build-pipeline smoke test for rskan PyO3 bindings.

Run:
    cd ~/projects/rskan/rskan-py && maturin develop --release
    uv run pytest tests/test_smoke.py
"""

import numpy as np
import pytest
import rskan


def test_module_imports():
    assert hasattr(rskan, "KanLayer")


def test_construct_kanlayer_with_init():
    layer = rskan.KanLayer(
        in_dim=3, out_dim=5, num=5, k=3,
        noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
        grid_range=(-1.0, 1.0), sp_trainable=True, sb_trainable=True,
        seed=1, device="cpu",
    )
    g = layer.grid()
    assert isinstance(g, np.ndarray)
    assert g.dtype == np.float32
    assert g.shape == (3, 5 + 1 + 2 * 3)


def test_construct_kanlayer_from_parts_validates_shapes():
    # Extended-grid knots = num + 1 + 2*k = 5 + 1 + 6 = 12. We pass 11 to force
    # a shape mismatch — KanLayerConfig::init_from_parts asserts on dims, which
    # the binding catches via std::panic::catch_unwind and re-raises as ValueError.
    grid = np.ones((3, 11), dtype=np.float32)         # wrong: should be 12
    coef = np.zeros((3, 5, 8), dtype=np.float32)
    sb   = np.ones((3, 5), dtype=np.float32)
    ss   = np.ones((3, 5), dtype=np.float32)
    mask = np.ones((3, 5), dtype=np.float32)
    with pytest.raises(Exception):
        rskan.KanLayer.from_parts(
            grid=grid, coef=coef, scale_base=sb, scale_sp=ss, mask=mask,
            in_dim=3, out_dim=5, num=5, k=3, seed=1, device="cpu",
        )
