//! PyO3 bindings for rskan. Forward + gradient extraction; no torch.autograd
//! integration in v1.

use pyo3::prelude::*;

mod layer;

#[pymodule]
fn _rskan_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<layer::PyKanLayer>()?;
    Ok(())
}
