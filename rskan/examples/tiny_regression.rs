//! Tiny regression smoke: fit y = sin(πx_0) + 0.3·cos(2πx_1) with a 2-layer KAN.
//!
//! Deterministic (seeded). Reports final loss; prints "PASS" if it drops below
//! a fixed threshold. Not a parity test — just a self-consistency sanity check
//! that the autodiff path actually converges.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use rskan::{Kan, KanConfig};

type B = Autodiff<NdArray<f32>>;

const SEED: u64 = 0xC0FFEE;
const BATCH: usize = 256;
const STEPS: usize = 400;
const LR: f64 = 1e-2;
const TARGET_LOSS: f32 = 5e-2;

fn sample_inputs(seed: u64) -> (Tensor<B, 2>, Tensor<B, 2>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = vec![0.0_f32; BATCH * 2];
    let mut y = vec![0.0_f32; BATCH];
    for b in 0..BATCH {
        let x0 = rng.gen::<f32>() * 2.0 - 1.0;
        let x1 = rng.gen::<f32>() * 2.0 - 1.0;
        x[b * 2] = x0;
        x[b * 2 + 1] = x1;
        y[b] = (std::f32::consts::PI * x0).sin() + 0.3 * (2.0 * std::f32::consts::PI * x1).cos();
    }
    let device = Default::default();
    let x_t = Tensor::from_data(TensorData::new(x, [BATCH, 2]), &device);
    let y_t = Tensor::from_data(TensorData::new(y, [BATCH, 1]), &device);
    (x_t, y_t)
}

fn main() {
    let device = Default::default();
    let mut model: Kan<B> = KanConfig::new(vec![2, 8, 1], SEED)
        .with_grid(5)
        .with_k(3)
        .with_noise_scale(0.3)
        .init(&device);
    let mut optim = AdamConfig::new().init();

    let (x, y_true) = sample_inputs(SEED ^ 1);

    let mut last_loss = f32::INFINITY;
    for step in 0..STEPS {
        let y_pred = model.forward(x.clone());
        let diff = y_pred - y_true.clone();
        let loss = diff.clone() * diff;
        let loss = loss.mean();
        last_loss = loss.clone().into_scalar();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads);

        if step % 50 == 0 || step == STEPS - 1 {
            println!("step {step:4}  loss = {last_loss:.6}");
        }
    }

    if last_loss < TARGET_LOSS {
        println!("PASS  final loss {last_loss:.6} < {TARGET_LOSS:.6}");
    } else {
        eprintln!("FAIL  final loss {last_loss:.6} >= {TARGET_LOSS:.6}");
        std::process::exit(1);
    }
}
