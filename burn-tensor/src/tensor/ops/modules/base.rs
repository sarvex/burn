use super::conv;
use crate::backend::Backend;

/// Gradient computed during the backward pass for each tensor used by [conv2d](ModuleOps::conv2d).
#[derive(new)]
pub struct Conv2dBackward<B: Backend> {
    pub x_grad: B::TensorPrimitive<4>,
    pub weights_grad: B::TensorPrimitive<4>,
    pub bias_grad: Option<B::TensorPrimitive<1>>,
}

pub trait ModuleOps<B: Backend> {
    fn embedding(
        weights: &B::TensorPrimitive<2>,
        indexes: &<B::IntegerBackend as Backend>::TensorPrimitive<2>,
    ) -> B::TensorPrimitive<3>;
    fn embedding_backward(
        weights: &B::TensorPrimitive<2>,
        output: &B::TensorPrimitive<3>,
        indexes: &<B::IntegerBackend as Backend>::TensorPrimitive<2>,
    ) -> B::TensorPrimitive<2>;
    /// Two dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      [batch_size, channels_in, height, width],
    /// weight: [channels_out, channels_in, kernel_size_1, kernel_size_2],
    /// bias:   [channels_out],
    fn conv2d(
        x: &B::TensorPrimitive<4>,
        weight: &B::TensorPrimitive<4>,
        bias: Option<&B::TensorPrimitive<1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> B::TensorPrimitive<4>;
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation.
    fn conv2d_backward(
        x: &B::TensorPrimitive<4>,
        weight: &B::TensorPrimitive<4>,
        bias: Option<&B::TensorPrimitive<1>>,
        stride: [usize; 2],
        output_grad: &B::TensorPrimitive<4>,
    ) -> Conv2dBackward<B> {
        conv::conv2d_backward(x, weight, bias, stride, output_grad)
    }
    /// One dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      [batch_size, channels_in, length],
    /// weight: [channels_out, channels_in, kernel_size],
    /// bias:   [channels_out],
    fn conv1d(
        x: &B::TensorPrimitive<3>,
        weight: &B::TensorPrimitive<3>,
        bias: Option<&B::TensorPrimitive<1>>,
        stride: usize,
        padding: usize,
    ) -> B::TensorPrimitive<3> {
        conv::conv1d_from_conv2d::<B>(x, weight, bias, stride, padding)
    }
}