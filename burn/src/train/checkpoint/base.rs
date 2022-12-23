use crate::module::{State, StateError, Module};
use burn_tensor::backend::Backend;

#[derive(Debug)]
pub enum CheckpointerError {
    IOError(std::io::Error),
    StateError(StateError),
}

pub trait Checkpointer<E> {
    fn save(&self, epoch: usize, state: State<E>) -> Result<(), CheckpointerError>;
    fn restore(&self, epoch: usize) -> Result<State<E>, CheckpointerError>;
}

pub struct TrainingCheckpointer<B: Backend> {
    checkpointer_model: Option<Box<dyn Checkpointer<B::Elem>>>,
    checkpointer_optimizer: Option<Box<dyn Checkpointer<B::Elem>>>,
}

impl<B: Backend> TrainingCheckpointer<B> {
    fn save<M: Module>(&self, epoch: usize, ) -> Result<(), CheckpointerError> {
        if let Some(checkpointer) = &self.checkpointer_model {
            checkpointer.save(epoch, self.model.state()).unwrap();
        }
        if let Some(checkpointer) = &self.checkpointer_optimizer {
            checkpointer
                .save(epoch, self.optim.state(&self.model))
                .unwrap();
        }
    }

    fn restore(&self, epoch: usize) -> Result<State<B::Elem>, CheckpointerError> {
        todo!()
    }
}
