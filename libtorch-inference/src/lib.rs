use ndarray::{ArrayD, IxDyn};
use std::path::Path;
use tch::{CModule, TchError, Tensor};
use ort::{init, Session, CUDAExecutionProvider, SessionInputs, SessionOutputs, Value, TensorElementType, IntoTensorElementType};
use std::sync::Arc;

pub type LibtorchResult<T> = Result<T, LibtorchError>;

#[derive(Debug)]
pub enum LibtorchError {
    InferenceError(TchError),
    OperationError(String),
}

impl From<TchError> for LibtorchError {
    fn from(other: TchError) -> Self {
        Self::InferenceError(other)
    }
}

pub struct LibtorchInferenceSession(Session);

impl LibtorchInferenceSession {
    pub fn from_path(model_path: impl AsRef<Path>) -> LibtorchResult<Self> {
        if !model_path.as_ref().exists() {
            return Err(LibtorchError::OperationError(format!(
                "Model file not found: `{}`",
                model_path.as_ref().display()
            )));
        }
        let session = create_inference_session(model_path)?;
        Ok(Self(session))
    }

    pub fn run(&self, inputs: &[Tensor]) -> LibtorchResult<LibtorchOutput> {
        let output = self.0.run(SessionInputs::from(inputs))?;
        Ok(LibtorchOutput(output.into()))
    }
}

pub struct LibtorchOutput(Vec<Tensor>);

impl LibtorchOutput {
    pub fn try_into_array<T: tch::kind::Element>(self) -> LibtorchResult<ArrayD<T>> {
        let tensor = self.0.into_iter().next().ok_or_else(|| {
            LibtorchError::OperationError("No tensor found in output".to_string())
        })?;
        let array = tensor.try_extract_tensor()?.to_owned();
        Ok(array)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use once_cell::sync::Lazy;
    use tch::{Device, Kind, Tensor};

    const SCRIPT_MODULE_PATH: &str = "./assets/model.pt";
    static INFERENCE_SESSION: Lazy<LibtorchInferenceSession> =
        Lazy::new(|| LibtorchInferenceSession::from_path(SCRIPT_MODULE_PATH).unwrap());

    #[test]
    fn test_basic() -> LibtorchResult<()> {
        let input = Tensor::rand([32], (Kind::Float, Device::Cpu));
        let output = INFERENCE_SESSION.run(&[input])?;
        let _array: ArrayD<f32> = output.try_into()?;
        Ok(())
    }

    #[test]
    fn test_with_ndarray_input() -> LibtorchResult<()> {
        let input = ndarray::Array1::<f32>::ones(32);
        let input_t: Tensor = input.try_into().unwrap();
        let output = INFERENCE_SESSION.run(&[input_t])?;
        let _array: ArrayD<f32> = output.try_into()?;
        Ok(())
    }
}

fn create_inference_session(model_path: &Path) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().with_device_id(0).build()?,
            // Add other execution providers as needed
        ])?
        .with_model_from_file(model_path)?
        .commit()?
}