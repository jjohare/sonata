/*
git clone --recursive https://github.com/mush42/sonata
cd sonata

wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
tar xf espeak-ng-data.tar.bz2
export ESPEAK_DATA_PATH=$(pwd)/espeak-ng-data

cargo run en_US-hfc_female-medium.onnx.json output.wav
*/

use sonata_synth::SonataSpeechSynthesizer;
use sonata_piper::from_config_path;
use std::path::Path;

fn init_ort_environment() -> anyhow::Result<()> {
    ort::init()
        .with_execution_providers([
            ort::CUDAExecutionProvider::default().with_device_id(0).build()?,
            ort::CPUExecutionProvider::default().build()?,
        ])?
        .commit()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    init_ort_environment()?;
    let config_path = std::env::args().nth(1).expect("Please specify config path");
    let output_path = std::env::args().nth(2).expect("Please specify output path");
    let text = "Hello! this is example with sonata".to_string();
    let voice = from_config_path(Path::new(&config_path))?;
    let synth = SonataSpeechSynthesizer::new(voice)?;
    synth
        .synthesize_to_file(Path::new(&output_path), text, None)
        .unwrap();
    Ok(())
}