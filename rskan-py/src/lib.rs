//! PyO3 bindings for rskan. Forward + gradient extraction; no torch.autograd
//! integration in v1.

use pyo3::prelude::*;

mod kan;
mod layer;

#[pymodule]
fn _rskan_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<layer::PyKanLayer>()?;
    m.add_class::<kan::PyKan>()?;
    Ok(())
}
