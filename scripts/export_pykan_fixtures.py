"""Export pykan KAN(Layer) modules + forward + backward outputs as .npy fixtures.

Run under DDR's uv venv:
    cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py

Writes to ~/projects/rskan/fixtures/.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from kan import KAN
from kan.KANLayer import KANLayer

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@dataclasses.dataclass
class LayerCase:
    name: str
    in_dim: int
    out_dim: int
    num: int
    k: int
    noise_scale: float = 0.5
    scale_base_mu: float = 0.0
    scale_base_sigma: float = 1.0
    scale_sp: float = 1.0
    grid_range: tuple = (-1.0, 1.0)
    sp_trainable: bool = True
    sb_trainable: bool = True
    weight_seed: int = 1
    batch: int = 16
    x_low: float = -0.9          # in-domain by default
    x_high: float = 0.9
    pykan_version: str = ""      # filled in at runtime


@dataclasses.dataclass
class KanCase:
    name: str
    widths: Sequence[int]
    grid: int = 5
    k: int = 3
    noise_scale: float = 0.3
    scale_base_mu: float = 0.0
    scale_base_sigma: float = 1.0
    scale_sp: float = 1.0
    grid_range: tuple = (-1.0, 1.0)
    weight_seed: int = 1
    batch: int = 16
    x_low: float = -0.9
    x_high: float = 0.9
    pykan_version: str = ""


def _x_seed(weight_seed: int) -> int:
    return weight_seed ^ 0xDEADBEEF


LAYER_CASES: list[LayerCase] = [
    LayerCase("kanlayer_i3_o5_k2_g3_s1",  in_dim=3,  out_dim=5,  num=3, k=2),
    LayerCase("kanlayer_i8_o8_k3_g5_s1",  in_dim=8,  out_dim=8,  num=5, k=3),
    LayerCase("kanlayer_i1_o1_k3_g5_s1",  in_dim=1,  out_dim=1,  num=5, k=3),
    LayerCase("kanlayer_i21_o21_k3_g5_s1", in_dim=21, out_dim=21, num=5, k=3),
    LayerCase("kanlayer_i3_o3_k3_g1_s1",   in_dim=3,  out_dim=3,  num=1, k=3),
    LayerCase(
        "kanlayer_i3_o3_k3_g5_s1_ood",
        in_dim=3, out_dim=3, num=5, k=3,
        x_low=-1.5, x_high=1.5,                    # OOD: outside grid_range
    ),
]

KAN_CASES: list[KanCase] = [
    KanCase("kan_w21x21x21_k3_g5_s1", widths=(21, 21, 21)),
    KanCase("kan_w8x8x8x8_k3_g5_s1",  widths=(8, 8, 8, 8)),
]


def _ensure_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _save(arr: torch.Tensor | np.ndarray, dst: Path) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    assert arr.dtype == np.float32, (
        f"expected float32 for {dst.name}, got {arr.dtype}; "
        "torch default dtype may have been mutated upstream"
    )
    np.save(dst, np.ascontiguousarray(arr))


def export_layer_case(c: LayerCase) -> dict:
    torch.manual_seed(c.weight_seed)
    layer = KANLayer(
        in_dim=c.in_dim, out_dim=c.out_dim,
        num=c.num, k=c.k,
        noise_scale=c.noise_scale,
        scale_base_mu=c.scale_base_mu, scale_base_sigma=c.scale_base_sigma,
        scale_sp=c.scale_sp,
        grid_range=list(c.grid_range),
        sp_trainable=c.sp_trainable, sb_trainable=c.sb_trainable,
        device="cpu",
    )
    case_dir = FIXTURES_DIR / c.name
    _ensure_clean(case_dir)

    _save(layer.grid.data,       case_dir / "grid.npy")
    _save(layer.coef.data,       case_dir / "coef.npy")
    _save(layer.scale_base.data, case_dir / "scale_base.npy")
    _save(layer.scale_sp.data,   case_dir / "scale_sp.npy")
    _save(layer.mask.data,       case_dir / "mask.npy")

    torch.manual_seed(_x_seed(c.weight_seed))
    x = torch.empty(c.batch, c.in_dim).uniform_(c.x_low, c.x_high)
    x.requires_grad_(True)

    y, _preacts, _postacts, _postspline = layer(x)
    _save(x, case_dir / "x.npy")
    _save(y, case_dir / "y.npy")

    y.sum().backward()
    _save(x.grad,                case_dir / "grad_x.npy")
    _save(layer.coef.grad,       case_dir / "grad_coef.npy")
    _save(layer.scale_base.grad, case_dir / "grad_scale_base.npy")
    _save(layer.scale_sp.grad,   case_dir / "grad_scale_sp.npy")

    params = dataclasses.asdict(c)
    params["x_seed"] = _x_seed(c.weight_seed)
    params["grid_range"] = list(params["grid_range"])
    params["pykan_version"] = _pykan_version()
    with (case_dir / "params.json").open("w") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    return {"name": c.name, "kind": "layer", "dir": c.name}


def export_kan_case(c: KanCase) -> dict:
    torch.manual_seed(c.weight_seed)
    # NOTE: KAN.__init__ (pykan 0.2.8) does NOT accept `scale_sp`; it hardcodes
    # scale_sp=1. internally when building each KANLayer. We pass base_fun='silu'
    # as a string (pykan 0.2.8 maps the string to torch.nn.SiLU()).
    model = KAN(
        width=list(c.widths),
        grid=c.grid, k=c.k,
        noise_scale=c.noise_scale,
        scale_base_mu=c.scale_base_mu, scale_base_sigma=c.scale_base_sigma,
        base_fun="silu",
        symbolic_enabled=True,
        affine_trainable=False,
        grid_eps=0.02,
        grid_range=list(c.grid_range),
        sp_trainable=True, sb_trainable=True,
        seed=c.weight_seed,
        save_act=False,
        sparse_init=False,
        auto_save=False,
        device="cpu",
    )
    case_dir = FIXTURES_DIR / c.name
    _ensure_clean(case_dir)

    # Per-layer parameter dumps + hooks for trajectory capture.
    trajectory: list[tuple[int, torch.Tensor]] = []

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            # KANLayer.forward returns (y, preacts, postacts, postspline);
            # keep only y for the trajectory.
            trajectory.append((idx, out[0].detach().clone()))
        return hook

    handles = []
    for l, layer in enumerate(model.act_fun):
        handles.append(layer.register_forward_hook(make_hook(l)))
        ldir = case_dir / f"layer_{l}"
        ldir.mkdir(parents=True, exist_ok=True)
        _save(layer.grid.data,       ldir / "grid.npy")
        _save(layer.coef.data,       ldir / "coef.npy")
        _save(layer.scale_base.data, ldir / "scale_base.npy")
        _save(layer.scale_sp.data,   ldir / "scale_sp.npy")
        _save(layer.mask.data,       ldir / "mask.npy")

    torch.manual_seed(_x_seed(c.weight_seed))
    in_dim = c.widths[0]
    x = torch.empty(c.batch, in_dim).uniform_(c.x_low, c.x_high)
    x.requires_grad_(True)

    y = model(x)
    _save(x, case_dir / "x.npy")
    _save(y, case_dir / "y.npy")

    # Sort the trajectory by layer index, save.
    trajectory.sort(key=lambda t: t[0])
    for idx, out in trajectory:
        _save(out, case_dir / f"trajectory_l{idx}.npy")

    y.sum().backward()
    _save(x.grad, case_dir / "grad_x.npy")
    for l, layer in enumerate(model.act_fun):
        ldir = case_dir / f"layer_{l}"
        _save(layer.coef.grad,       ldir / "grad_coef.npy")
        _save(layer.scale_base.grad, ldir / "grad_scale_base.npy")
        _save(layer.scale_sp.grad,   ldir / "grad_scale_sp.npy")

    for h in handles:
        h.remove()

    params = dataclasses.asdict(c)
    params["x_seed"] = _x_seed(c.weight_seed)
    params["widths"] = list(params["widths"])
    params["grid_range"] = list(params["grid_range"])
    params["pykan_version"] = _pykan_version()
    with (case_dir / "params.json").open("w") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    return {"name": c.name, "kind": "kan", "dir": c.name}


def _pykan_version() -> str:
    try:
        import kan
        v = getattr(kan, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    # Fall back to package metadata (pykan 0.2.8 has no kan.__version__).
    try:
        from importlib.metadata import version
        return version("pykan")
    except Exception:
        return "unknown"


def main() -> None:
    torch.set_default_dtype(torch.float32)
    EXPECTED_PYKAN = "0.2.8"
    detected = _pykan_version()
    assert detected == EXPECTED_PYKAN, (
        f"fixtures are pinned to pykan {EXPECTED_PYKAN}, but {detected} is installed; "
        "either install the expected version or bump EXPECTED_PYKAN and re-validate parity"
    )
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for c in LAYER_CASES:
        manifest.append(export_layer_case(c))
    for c in KAN_CASES:
        manifest.append(export_kan_case(c))
    with (FIXTURES_DIR / "manifest.json").open("w") as f:
        json.dump(
            {
                "schema_version": 1,
                "pykan_version": _pykan_version(),
                "cases": manifest,
            },
            f, indent=2, sort_keys=True,
        )
    print(f"Exported {len(manifest)} fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
