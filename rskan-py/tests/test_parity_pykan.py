"""Python-side parity sweep against pykan.

Runs only if `pykan` is importable in the active venv. Skipped otherwise.

Run:
    cd ~/projects/rskan/rskan-py && maturin develop --release
    cd ~/projects/ddr && uv run pytest ~/projects/rskan/rskan-py/tests/test_parity_pykan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    from kan.KANLayer import KANLayer
    HAVE_PYKAN = True
except Exception:
    HAVE_PYKAN = False

import rskan

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

pytestmark = pytest.mark.skipif(not HAVE_PYKAN, reason="pykan not installed in this env")


def _list_layer_cases() -> list[str]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    return [c["name"] for c in manifest["cases"] if c["kind"] == "layer"]


def _build_pykan_from_fixture(case_dir: Path):
    params = json.loads((case_dir / "params.json").read_text())
    layer = KANLayer(
        in_dim=params["in_dim"], out_dim=params["out_dim"],
        num=params["num"], k=params["k"],
        noise_scale=params["noise_scale"],
        scale_base_mu=params["scale_base_mu"],
        scale_base_sigma=params["scale_base_sigma"],
        scale_sp=params["scale_sp"],
        grid_range=list(params["grid_range"]),
        sp_trainable=params["sp_trainable"],
        sb_trainable=params["sb_trainable"],
        device="cpu",
    )
    with torch.no_grad():
        layer.grid.copy_(torch.from_numpy(np.load(case_dir / "grid.npy")))
        layer.coef.copy_(torch.from_numpy(np.load(case_dir / "coef.npy")))
        layer.scale_base.copy_(torch.from_numpy(np.load(case_dir / "scale_base.npy")))
        layer.scale_sp.copy_(torch.from_numpy(np.load(case_dir / "scale_sp.npy")))
        layer.mask.copy_(torch.from_numpy(np.load(case_dir / "mask.npy")))
    return layer, params


def _build_rskan_from_fixture(case_dir: Path):
    params = json.loads((case_dir / "params.json").read_text())
    return rskan.KanLayer.from_parts(
        grid=np.load(case_dir / "grid.npy"),
        coef=np.load(case_dir / "coef.npy"),
        scale_base=np.load(case_dir / "scale_base.npy"),
        scale_sp=np.load(case_dir / "scale_sp.npy"),
        mask=np.load(case_dir / "mask.npy"),
        in_dim=params["in_dim"], out_dim=params["out_dim"],
        num=params["num"], k=params["k"],
        seed=params["weight_seed"],
        noise_scale=params["noise_scale"],
        scale_base_mu=params["scale_base_mu"],
        scale_base_sigma=params["scale_base_sigma"],
        scale_sp_arg=params["scale_sp"],
        grid_range=tuple(params["grid_range"]),
        sp_trainable=params["sp_trainable"],
        sb_trainable=params["sb_trainable"],
        device="cpu",
    )


@pytest.mark.parametrize("case", _list_layer_cases())
def test_forward_and_grad_matches_pykan(case):
    case_dir = FIXTURES / case
    rskan_layer = _build_rskan_from_fixture(case_dir)
    pykan_layer, _params = _build_pykan_from_fixture(case_dir)

    x = np.load(case_dir / "x.npy")

    # rskan forward + grads.
    y_rskan, grads = rskan_layer.forward_with_grad(x)
    assert y_rskan.shape == np.load(case_dir / "y.npy").shape

    # pykan forward + backward.
    x_t = torch.tensor(x, requires_grad=True)
    y_t, *_ = pykan_layer(x_t)
    y_t.sum().backward()

    np.testing.assert_allclose(
        y_rskan, y_t.detach().numpy(),
        atol=1e-5, rtol=1e-4, err_msg=f"{case}: forward y",
    )
    np.testing.assert_allclose(
        grads["x"], x_t.grad.numpy(),
        atol=1e-4, rtol=1e-3, err_msg=f"{case}: grad_x",
    )
    np.testing.assert_allclose(
        grads["coef"], pykan_layer.coef.grad.numpy(),
        atol=1e-4, rtol=1e-3, err_msg=f"{case}: grad_coef",
    )
    np.testing.assert_allclose(
        grads["scale_base"], pykan_layer.scale_base.grad.numpy(),
        atol=1e-4, rtol=1e-3, err_msg=f"{case}: grad_scale_base",
    )
    np.testing.assert_allclose(
        grads["scale_sp"], pykan_layer.scale_sp.grad.numpy(),
        atol=1e-4, rtol=1e-3, err_msg=f"{case}: grad_scale_sp",
    )


def test_forward_with_grad_explicit_grad_y():
    layer = rskan.KanLayer(in_dim=2, out_dim=3, num=5, k=3, seed=5, device="cpu")
    x = np.random.RandomState(0).uniform(-0.5, 0.5, (8, 2)).astype(np.float32)
    grad_y = np.ones((8, 3), dtype=np.float32)
    y_a, grads_a = layer.forward_with_grad(x)                     # implicit ones
    y_b, grads_b = layer.forward_with_grad(x, grad_y=grad_y)      # explicit ones
    np.testing.assert_allclose(y_a, y_b, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(grads_a["x"], grads_b["x"], atol=1e-7, rtol=1e-6)


def test_forward_with_grad_validates_grad_y_shape():
    layer = rskan.KanLayer(in_dim=2, out_dim=3, num=5, k=3, seed=5, device="cpu")
    x = np.zeros((4, 2), dtype=np.float32)
    bad = np.zeros((4, 5), dtype=np.float32)                      # wrong out_dim
    with pytest.raises(Exception):
        layer.forward_with_grad(x, grad_y=bad)
