use burn::optim::{decay::WeightDecayConfig, momentum::MomentumConfig};
use text_generation::{training::ExperimentConfig, C4Dataset};

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<burn::tensor::f16>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(1024, 4096, 16, 8),
        burn::optim::SgdConfig::new()
            .with_learning_rate(1.0e-4)
            .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
            .with_momentum(Some(MomentumConfig::new().with_nesterov(true))),
    );

    text_generation::training::train::<Backend, C4Dataset>(
        vec![
            burn_tch::TchDevice::Cuda(0),
            burn_tch::TchDevice::Cuda(1),
            burn_tch::TchDevice::Cuda(2),
            burn_tch::TchDevice::Cuda(3),
        ],
        C4Dataset::train(),
        C4Dataset::test(),
        config,
        "/tmp/text-generation",
    );
}
