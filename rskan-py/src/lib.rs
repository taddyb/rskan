//! PyO3 bindings for rskan. Filled in starting Task 18.

use pyo3::prelude::*;

#[pymodule]
fn _rskan_py(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
