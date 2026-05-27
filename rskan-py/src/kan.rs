//! `PyKan` — multi-layer Python wrapper.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rskan::{Kan, KanConfig};

type B = Autodiff<NdArray<f32>>;
type Dev = NdArrayDevice;

#[pyclass(name = "Kan", module = "rskan")]
pub struct PyKan {
    inner: Kan<B>,
    device: Dev,
}

fn check_device(device: &str) -> PyResult<()> {
    if device == "cpu" {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "device {device:?} not supported in this build"
        )))
    }
}

fn nd2_to_tensor(arr: Array2<f32>, device: &Dev) -> Tensor<B, 2> {
    let (r, c) = (arr.shape()[0], arr.shape()[1]);
    let standard = arr.as_standard_layout().to_owned();
    let (vec, _) = standard.into_raw_vec_and_offset();
    Tensor::from_data(TensorData::new(vec, [r, c]), device)
}

fn tensor2_to_pyarray<'py, BB: Backend>(
    py: Python<'py>,
    t: Tensor<BB, 2>,
) -> Bound<'py, PyArray2<f32>> {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let arr = Array2::from_shape_vec((dims[0], dims[1]), slice.to_vec()).unwrap();
    arr.into_pyarray_bound(py)
}

fn tensor3_to_pyarray<'py, BB: Backend>(
    py: Python<'py>,
    t: Tensor<BB, 3>,
) -> Bound<'py, PyArray3<f32>> {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), slice.to_vec()).unwrap();
    arr.into_pyarray_bound(py)
}

#[pymethods]
impl PyKan {
    #[new]
    #[pyo3(signature = (
        *, widths, seed,
        grid=3, k=3, noise_scale=0.3,
        scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
        grid_range=(-1.0, 1.0), sp_trainable=true, sb_trainable=true,
        device="cpu",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        widths: Vec<usize>,
        seed: u64,
        grid: usize,
        k: usize,
        noise_scale: f64,
        scale_base_mu: f64,
        scale_base_sigma: f64,
        scale_sp: f64,
        grid_range: (f64, f64),
        sp_trainable: bool,
        sb_trainable: bool,
        device: &str,
    ) -> PyResult<Self> {
        check_device(device)?;
        if widths.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "widths must have at least 2 elements, got {}",
                widths.len()
            )));
        }
        let dev: Dev = Default::default();
        let cfg = KanConfig::new(widths, seed)
            .with_grid(grid)
            .with_k(k)
            .with_noise_scale(noise_scale)
            .with_scale_base_mu(scale_base_mu)
            .with_scale_base_sigma(scale_base_sigma)
            .with_scale_sp(scale_sp)
            .with_grid_range([grid_range.0, grid_range.1])
            .with_sp_trainable(sp_trainable)
            .with_sb_trainable(sb_trainable);

        let inner = cfg.init::<B>(&dev);
        Ok(PyKan { inner, device: dev })
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_view = x.as_array();
        let in_dim_expected = self.inner.layers[0].grid.val().dims()[0];
        if x_view.shape()[1] != in_dim_expected {
            return Err(PyValueError::new_err(format!(
                "x in_dim {} != model in_dim {in_dim_expected}",
                x_view.shape()[1]
            )));
        }
        let x_t = nd2_to_tensor(x_view.to_owned(), &self.device);
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
        let in_dim_expected = self.inner.layers[0].grid.val().dims()[0];
        if in_dim != in_dim_expected {
            return Err(PyValueError::new_err(format!(
                "x in_dim {in_dim} != model in_dim {in_dim_expected}"
            )));
        }
        let out_dim = self.inner.layers.last().unwrap().coef.val().dims()[1];

        let x_t = nd2_to_tensor(x_view.to_owned(), &self.device).require_grad();
        let y_t = self.inner.forward(x_t.clone());

        let loss = if let Some(gy) = grad_y {
            let gy_view = gy.as_array();
            let gy_shape = gy_view.shape();
            if gy_shape[0] != batch || gy_shape[1] != out_dim {
                return Err(PyValueError::new_err(format!(
                    "grad_y shape {:?} != y shape [{batch}, {out_dim}]",
                    gy_view.shape()
                )));
            }
            (y_t.clone() * nd2_to_tensor(gy_view.to_owned(), &self.device)).sum()
        } else {
            y_t.clone().sum()
        };
        let grads = loss.backward();

        // Per-layer grads list.
        let layers_list = PyList::empty_bound(py);
        for layer in &self.inner.layers {
            let gcoef = layer
                .coef
                .val()
                .grad(&grads)
                .ok_or_else(|| PyTypeError::new_err("layer coef.grad is None"))?;
            let gsb = layer
                .scale_base
                .val()
                .grad(&grads)
                .ok_or_else(|| PyTypeError::new_err("layer scale_base.grad is None"))?;
            let gsp = layer
                .scale_sp
                .val()
                .grad(&grads)
                .ok_or_else(|| PyTypeError::new_err("layer scale_sp.grad is None"))?;

            let d = PyDict::new_bound(py);
            d.set_item("coef", tensor3_to_pyarray(py, gcoef))?;
            d.set_item("scale_base", tensor2_to_pyarray(py, gsb))?;
            d.set_item("scale_sp", tensor2_to_pyarray(py, gsp))?;
            layers_list.append(d)?;
        }

        let gx = x_t
            .grad(&grads)
            .ok_or_else(|| PyTypeError::new_err("x.grad is None"))?;

        let top = PyDict::new_bound(py);
        top.set_item("x", tensor2_to_pyarray(py, gx))?;
        top.set_item("layers", layers_list)?;

        let y_np = tensor2_to_pyarray(py, y_t);
        Ok(PyTuple::new_bound(py, &[y_np.into_py(py), top.into_py(py)]).into())
    }

    fn num_layers(&self) -> usize {
        self.inner.layers.len()
    }
}
