use burn::optim::decay::WeightDecayConfig;
use text_generation::{training::ExperimentConfig, DbPediaDataset};

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<burn::tensor::f16>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(1024, 4096, 16, 8),
        burn::optim::AdamConfig::new(1.0e-4)
            .with_epsilon(1e-4)
            .with_weight_decay(Some(WeightDecayConfig::new(5e-6))),
    );

    text_generation::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            burn_tch::TchDevice::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
