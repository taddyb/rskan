//! `PyKanLayer` — Python-facing KanLayer with NdArray-Autodiff backend internally.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
#[allow(unused_imports)]
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
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

fn tensor2_to_pyarray<'py>(py: Python<'py>, t: Tensor<B, 2>) -> Bound<'py, PyArray2<f32>> {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let arr = Array2::from_shape_vec((dims[0], dims[1]), slice.to_vec()).unwrap();
    arr.into_pyarray_bound(py)
}
fn tensor3_to_pyarray<'py>(py: Python<'py>, t: Tensor<B, 3>) -> Bound<'py, PyArray3<f32>> {
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
}
