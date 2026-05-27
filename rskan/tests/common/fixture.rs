//! Fixture loader. Reads pykan-exported .npy + params.json from `fixtures/`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::{Array2, Array3, ArrayD};
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LayerParams {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub num: usize,
    pub k: usize,
    pub noise_scale: f64,
    pub scale_base_mu: f64,
    pub scale_base_sigma: f64,
    pub scale_sp: f64,
    pub grid_range: [f64; 2],
    pub sp_trainable: bool,
    pub sb_trainable: bool,
    pub weight_seed: u64,
    pub x_seed: u64,
    pub batch: usize,
}

#[derive(Debug, Deserialize)]
pub struct KanParams {
    pub name: String,
    pub widths: Vec<usize>,
    pub grid: usize,
    pub k: usize,
    pub noise_scale: f64,
    pub scale_base_mu: f64,
    pub scale_base_sigma: f64,
    pub scale_sp: f64,
    pub grid_range: [f64; 2],
    pub weight_seed: u64,
    pub x_seed: u64,
    pub batch: usize,
}

#[derive(Debug, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub kind: String, // "layer" or "kan"
    pub dir: String,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub cases: Vec<ManifestEntry>,
    // schema_version, pykan_version present in JSON but loader can ignore them
    // via serde's default behavior of ignoring unknown fields.
}

pub fn fixtures_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = rskan/rskan; the workspace root is its parent.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir).parent().unwrap().join("fixtures")
}

pub fn load_manifest() -> Manifest {
    let path = fixtures_root().join("manifest.json");
    let f = File::open(&path)
        .unwrap_or_else(|e| panic!("could not open {}: {e}", path.display()));
    serde_json::from_reader(BufReader::new(f)).expect("parse manifest.json")
}

fn read_npy_dyn(path: &Path) -> ArrayD<f32> {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    ArrayD::<f32>::read_npy(f)
        .unwrap_or_else(|e| panic!("read_npy {}: {e}", path.display()))
}

pub fn read_npy_2d(path: &Path) -> Array2<f32> {
    let dyn_arr = read_npy_dyn(path);
    let shape = dyn_arr.shape().to_vec();
    assert_eq!(
        shape.len(),
        2,
        "{}: expected 2D, got {:?}",
        path.display(),
        shape
    );
    dyn_arr.into_dimensionality::<ndarray::Ix2>().unwrap()
}

pub fn read_npy_3d(path: &Path) -> Array3<f32> {
    let dyn_arr = read_npy_dyn(path);
    let shape = dyn_arr.shape().to_vec();
    assert_eq!(
        shape.len(),
        3,
        "{}: expected 3D, got {:?}",
        path.display(),
        shape
    );
    dyn_arr.into_dimensionality::<ndarray::Ix3>().unwrap()
}

pub fn read_params_layer(dir: &Path) -> LayerParams {
    let f = File::open(dir.join("params.json"))
        .unwrap_or_else(|e| panic!("open params.json in {}: {e}", dir.display()));
    serde_json::from_reader(BufReader::new(f)).expect("parse params.json")
}

pub fn read_params_kan(dir: &Path) -> KanParams {
    let f = File::open(dir.join("params.json"))
        .unwrap_or_else(|e| panic!("open params.json in {}: {e}", dir.display()));
    serde_json::from_reader(BufReader::new(f)).expect("parse params.json")
}

/// One bare-KANLayer fixture, materialized on backend `B`.
pub struct LayerFixture<B: Backend> {
    pub params: LayerParams,
    pub grid: Tensor<B, 2>,
    pub coef: Tensor<B, 3>,
    pub scale_base: Tensor<B, 2>,
    pub scale_sp: Tensor<B, 2>,
    pub mask: Tensor<B, 2>,
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 2>,
    pub grad_x: Tensor<B, 2>,
    pub grad_coef: Tensor<B, 3>,
    pub grad_scale_base: Tensor<B, 2>,
    pub grad_scale_sp: Tensor<B, 2>,
}

fn nd2_to_tensor<B: Backend>(arr: Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let (r, c) = (arr.shape()[0], arr.shape()[1]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [r, c]);
    Tensor::from_data(data, device)
}
fn nd3_to_tensor<B: Backend>(arr: Array3<f32>, device: &B::Device) -> Tensor<B, 3> {
    let (d0, d1, d2) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [d0, d1, d2]);
    Tensor::from_data(data, device)
}

pub fn load_layer_fixture<B: Backend>(case: &str, device: &B::Device) -> LayerFixture<B> {
    let dir = fixtures_root().join(case);
    let params = read_params_layer(&dir);

    LayerFixture {
        params,
        grid: nd2_to_tensor(read_npy_2d(&dir.join("grid.npy")), device),
        coef: nd3_to_tensor(read_npy_3d(&dir.join("coef.npy")), device),
        scale_base: nd2_to_tensor(read_npy_2d(&dir.join("scale_base.npy")), device),
        scale_sp: nd2_to_tensor(read_npy_2d(&dir.join("scale_sp.npy")), device),
        mask: nd2_to_tensor(read_npy_2d(&dir.join("mask.npy")), device),
        x: nd2_to_tensor(read_npy_2d(&dir.join("x.npy")), device),
        y: nd2_to_tensor(read_npy_2d(&dir.join("y.npy")), device),
        grad_x: nd2_to_tensor(read_npy_2d(&dir.join("grad_x.npy")), device),
        grad_coef: nd3_to_tensor(read_npy_3d(&dir.join("grad_coef.npy")), device),
        grad_scale_base: nd2_to_tensor(
            read_npy_2d(&dir.join("grad_scale_base.npy")),
            device,
        ),
        grad_scale_sp: nd2_to_tensor(
            read_npy_2d(&dir.join("grad_scale_sp.npy")),
            device,
        ),
    }
}

/// Multi-layer Kan fixture: per-layer tensors plus the trajectory.
pub struct KanLayerSlice<B: Backend> {
    pub grid: Tensor<B, 2>,
    pub coef: Tensor<B, 3>,
    pub scale_base: Tensor<B, 2>,
    pub scale_sp: Tensor<B, 2>,
    pub mask: Tensor<B, 2>,
    pub grad_coef: Tensor<B, 3>,
    pub grad_scale_base: Tensor<B, 2>,
    pub grad_scale_sp: Tensor<B, 2>,
}

pub struct KanFixture<B: Backend> {
    pub params: KanParams,
    pub layers: Vec<KanLayerSlice<B>>,
    pub trajectory: Vec<Tensor<B, 2>>,
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 2>,
    pub grad_x: Tensor<B, 2>,
}

pub fn load_kan_fixture<B: Backend>(case: &str, device: &B::Device) -> KanFixture<B> {
    let dir = fixtures_root().join(case);
    let params = read_params_kan(&dir);
    let num_layers = params.widths.len() - 1;

    let mut layers = Vec::with_capacity(num_layers);
    let mut trajectory = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let ldir = dir.join(format!("layer_{l}"));
        layers.push(KanLayerSlice {
            grid: nd2_to_tensor(read_npy_2d(&ldir.join("grid.npy")), device),
            coef: nd3_to_tensor(read_npy_3d(&ldir.join("coef.npy")), device),
            scale_base: nd2_to_tensor(read_npy_2d(&ldir.join("scale_base.npy")), device),
            scale_sp: nd2_to_tensor(read_npy_2d(&ldir.join("scale_sp.npy")), device),
            mask: nd2_to_tensor(read_npy_2d(&ldir.join("mask.npy")), device),
            grad_coef: nd3_to_tensor(read_npy_3d(&ldir.join("grad_coef.npy")), device),
            grad_scale_base: nd2_to_tensor(
                read_npy_2d(&ldir.join("grad_scale_base.npy")),
                device,
            ),
            grad_scale_sp: nd2_to_tensor(
                read_npy_2d(&ldir.join("grad_scale_sp.npy")),
                device,
            ),
        });
        trajectory.push(nd2_to_tensor(
            read_npy_2d(&dir.join(format!("trajectory_l{l}.npy"))),
            device,
        ));
    }

    KanFixture {
        params,
        layers,
        trajectory,
        x: nd2_to_tensor(read_npy_2d(&dir.join("x.npy")), device),
        y: nd2_to_tensor(read_npy_2d(&dir.join("y.npy")), device),
        grad_x: nd2_to_tensor(read_npy_2d(&dir.join("grad_x.npy")), device),
    }
}
