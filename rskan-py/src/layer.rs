//! `PyKanLayer` — Python-facing KanLayer with NdArray-Autodiff backend internally.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rskan::{KanLayer, KanLayerConfig};

type B = Autodiff<NdArray<f32>>;
type Dev = NdArrayDevice;

#[pyclass(name = "KanLayer", module = "rskan")]
pub struct PyKanLayer {
    inner: KanLayer<B>,
    #[allow(dead_code)]
    device: Dev,
}

impl PyKanLayer {
    #[allow(clippy::too_many_arguments)]
    fn cfg_from_kwargs(
        in_dim: usize,
        out_dim: usize,
        num: usize,
        k: usize,
        noise_scale: f64,
        scale_base_mu: f64,
        scale_base_sigma: f64,
        scale_sp: f64,
        grid_range: (f64, f64),
        sp_trainable: bool,
        sb_trainable: bool,
        seed: u64,
    ) -> KanLayerConfig {
        KanLayerConfig::new(in_dim, out_dim, seed)
            .with_num(num)
            .with_k(k)
            .with_noise_scale(noise_scale)
            .with_scale_base_mu(scale_base_mu)
            .with_scale_base_sigma(scale_base_sigma)
            .with_scale_sp(scale_sp)
            .with_grid_range([grid_range.0, grid_range.1])
            .with_sp_trainable(sp_trainable)
            .with_sb_trainable(sb_trainable)
    }
}

fn check_device(device: &str) -> PyResult<()> {
    if device == "cpu" {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "device {device:?} not supported in this build; only \"cpu\" available without the cuda feature"
        )))
    }
}

fn nd2_from(arr: PyReadonlyArray2<'_, f32>) -> PyResult<Array2<f32>> {
    Ok(arr.as_array().to_owned())
}
fn nd3_from(arr: PyReadonlyArray3<'_, f32>) -> PyResult<Array3<f32>> {
    Ok(arr.as_array().to_owned())
}

fn nd2_to_tensor(arr: Array2<f32>, device: &Dev) -> Tensor<B, 2> {
    let (r, c) = (arr.shape()[0], arr.shape()[1]);
    let standard = arr.as_standard_layout().to_owned();
    let (vec, _) = standard.into_raw_vec_and_offset();
    Tensor::from_data(TensorData::new(vec, [r, c]), device)
}
fn nd3_to_tensor(arr: Array3<f32>, device: &Dev) -> Tensor<B, 3> {
    let (d0, d1, d2) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let standard = arr.as_standard_layout().to_owned();
    let (vec, _) = standard.into_raw_vec_and_offset();
    Tensor::from_data(TensorData::new(vec, [d0, d1, d2]), device)
}

fn tensor2_to_pyarray<'py, BB: Backend>(py: Python<'py>, t: Tensor<BB, 2>) -> Bound<'py, PyArray2<f32>> {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let arr = Array2::from_shape_vec((dims[0], dims[1]), slice.to_vec()).unwrap();
    arr.into_pyarray_bound(py)
}
fn tensor3_to_pyarray<'py, BB: Backend>(py: Python<'py>, t: Tensor<BB, 3>) -> Bound<'py, PyArray3<f32>> {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), slice.to_vec()).unwrap();
    arr.into_pyarray_bound(py)
}

#[pymethods]
impl PyKanLayer {
    #[new]
    #[pyo3(signature = (
        *, in_dim, out_dim, seed,
        num=5, k=3, noise_scale=0.5,
        scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
        grid_range=(-1.0, 1.0), sp_trainable=true, sb_trainable=true,
        device="cpu",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_dim: usize, out_dim: usize, seed: u64,
        num: usize, k: usize, noise_scale: f64,
        scale_base_mu: f64, scale_base_sigma: f64, scale_sp: f64,
        grid_range: (f64, f64), sp_trainable: bool, sb_trainable: bool,
        device: &str,
    ) -> PyResult<Self> {
        check_device(device)?;
        let dev = Default::default();
        let cfg = Self::cfg_from_kwargs(
            in_dim, out_dim, num, k, noise_scale,
            scale_base_mu, scale_base_sigma, scale_sp,
            grid_range, sp_trainable, sb_trainable, seed,
        );
        let inner = cfg.init::<B>(&dev);
        Ok(PyKanLayer { inner, device: dev })
    }

    #[staticmethod]
    #[pyo3(signature = (
        *, grid, coef, scale_base, scale_sp, mask,
        in_dim, out_dim, num, k, seed,
        noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0,
        scale_sp_arg=1.0, grid_range=(-1.0, 1.0),
        sp_trainable=true, sb_trainable=true, device="cpu",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        _py: Python<'_>,
        grid: PyReadonlyArray2<f32>, coef: PyReadonlyArray3<f32>,
        scale_base: PyReadonlyArray2<f32>, scale_sp: PyReadonlyArray2<f32>,
        mask: PyReadonlyArray2<f32>,
        in_dim: usize, out_dim: usize, num: usize, k: usize, seed: u64,
        noise_scale: f64, scale_base_mu: f64, scale_base_sigma: f64,
        scale_sp_arg: f64, grid_range: (f64, f64),
        sp_trainable: bool, sb_trainable: bool, device: &str,
    ) -> PyResult<Self> {
        check_device(device)?;
        let dev = Default::default();
        let cfg = Self::cfg_from_kwargs(
            in_dim, out_dim, num, k, noise_scale,
            scale_base_mu, scale_base_sigma, scale_sp_arg,
            grid_range, sp_trainable, sb_trainable, seed,
        );

        let g = nd2_to_tensor(nd2_from(grid)?,           &dev);
        let c = nd3_to_tensor(nd3_from(coef)?,           &dev);
        let s = nd2_to_tensor(nd2_from(scale_base)?,     &dev);
        let p = nd2_to_tensor(nd2_from(scale_sp)?,       &dev);
        let m = nd2_to_tensor(nd2_from(mask)?,           &dev);

        // The KanLayer asserts shapes internally; convert panic → Python exception.
        let inner = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cfg.init_from_parts::<B>(&dev, g, c, s, p, m)
        }))
        .map_err(|e| {
            let msg = if let Some(s) = e.downcast_ref::<&'static str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "init_from_parts panicked".to_string()
            };
            PyValueError::new_err(msg)
        })?;

        Ok(PyKanLayer { inner, device: dev })
    }

    fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        tensor2_to_pyarray(py, self.inner.grid.val())
    }
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        tensor3_to_pyarray(py, self.inner.coef.val())
    }
    fn scale_base<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        tensor2_to_pyarray(py, self.inner.scale_base.val())
    }
    fn scale_sp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        tensor2_to_pyarray(py, self.inner.scale_sp.val())
    }
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        tensor2_to_pyarray(py, self.inner.mask.val())
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_view = x.as_array();
        let (_batch, in_dim) = (x_view.shape()[0], x_view.shape()[1]);
        let expected_in = self.inner.grid.val().dims()[0];
        if in_dim != expected_in {
            return Err(PyValueError::new_err(format!(
                "x in_dim {in_dim} != layer in_dim {expected_in}"
            )));
        }
        let x_arr: Array2<f32> = x_view.to_owned();
        let x_t = nd2_to_tensor(x_arr, &self.device);
        let y_t = self.inner.forward(x_t);
        Ok(tensor2_to_pyarray(py, y_t))
    }

    #[pyo3(signature = (x, *, grad_y=None))]
    fn forward_with_grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        grad_y: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Py<PyTuple>> {
        let x_view = x.as_array();
        let (batch, in_dim) = (x_view.shape()[0], x_view.shape()[1]);
        let expected_in = self.inner.grid.val().dims()[0];
        if in_dim != expected_in {
            return Err(PyValueError::new_err(format!(
                "x in_dim {in_dim} != layer in_dim {expected_in}"
            )));
        }
        let out_dim = self.inner.coef.val().dims()[1];

        // Marshal x and (optional) grad_y into Burn tensors.
        let x_arr: Array2<f32> = x_view.to_owned();
        let x_t = nd2_to_tensor(x_arr, &self.device).require_grad();
        let y_t: Tensor<B, 2> = self.inner.forward(x_t.clone());

        // Compose the scalar loss whose gradient w.r.t. each leaf equals grad_y (or ones).
        let loss = if let Some(gy) = grad_y {
            let gy_view = gy.as_array();
            let gy_shape = gy_view.shape();
            if gy_shape[0] != batch || gy_shape[1] != out_dim {
                return Err(PyValueError::new_err(format!(
                    "grad_y shape {:?} != y shape [{batch}, {out_dim}]",
                    gy_shape
                )));
            }
            let gy_t = nd2_to_tensor(gy_view.to_owned(), &self.device);
            (y_t.clone() * gy_t).sum()
        } else {
            y_t.clone().sum()
        };

        let grads = loss.backward();

        let gx = x_t
            .grad(&grads)
            .ok_or_else(|| PyTypeError::new_err("x.grad is None — forward path may not be differentiable"))?;
        let gcoef = self
            .inner
            .coef
            .val()
            .grad(&grads)
            .ok_or_else(|| PyTypeError::new_err("coef.grad is None"))?;
        let gsb = self
            .inner
            .scale_base
            .val()
            .grad(&grads)
            .ok_or_else(|| PyTypeError::new_err("scale_base.grad is None"))?;
        let gsp = self
            .inner
            .scale_sp
            .val()
            .grad(&grads)
            .ok_or_else(|| PyTypeError::new_err("scale_sp.grad is None"))?;

        let y_np = tensor2_to_pyarray(py, y_t);
        let gx_np = tensor2_to_pyarray(py, gx);
        let gcoef_np = tensor3_to_pyarray(py, gcoef);
        let gsb_np = tensor2_to_pyarray(py, gsb);
        let gsp_np = tensor2_to_pyarray(py, gsp);

        let dict = PyDict::new_bound(py);
        dict.set_item("x", gx_np)?;
        dict.set_item("coef", gcoef_np)?;
        dict.set_item("scale_base", gsb_np)?;
        dict.set_item("scale_sp", gsp_np)?;

        Ok(PyTuple::new_bound(py, &[y_np.into_py(py), dict.into_py(py)]).into())
    }
}
