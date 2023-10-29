use espeak_phonemizer::text_to_phonemes;
use libtashkeel_base::do_tashkeel;
use ndarray::{Array, Array1, Array2, ArrayView, CowArray, Dim, IxDynImpl};
use ndarray_stats::QuantileExt;
use once_cell::sync::{Lazy, OnceCell};
use ort::{tensor::OrtOwnedTensor, Environment, GraphOptimizationLevel, SessionBuilder, Value};
use piper_core::{
    Phonemes, PiperError, PiperModel, PiperResult, PiperWaveInfo, PiperWaveResult,
    PiperWaveSamples, RawWaveSamples,
};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

const MAX_WAV_VALUE: f32 = 32767.0;
const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

#[allow(dead_code)]
static CPU_COUNT: Lazy<i16> = Lazy::new(|| num_cpus::get().try_into().unwrap_or(4));

#[inline(always)]
fn reversed_mapping<K, V>(input: &HashMap<K, V>) -> HashMap<V, K>
where
    K: ToOwned<Owned = K>,
    V: ToOwned<Owned = V> + std::hash::Hash + std::cmp::Eq,
{
    HashMap::from_iter(input.iter().map(|(k, v)| (v.to_owned(), k.to_owned())))
}

#[inline(always)]
fn audio_float_to_i16(audio_f32: ArrayView<f32, Dim<IxDynImpl>>) -> PiperResult<RawWaveSamples> {
    if audio_f32.is_empty() {
        return Ok(Default::default());
    }
    let Ok(min_audio_value) = audio_f32.min() else {
        return Err(PiperError::OperationError(
            "Invalid output from model inference.".to_string(),
        ));
    };
    let Ok(max_audio_value) = audio_f32.max() else {
        return Err(PiperError::OperationError(
            "Invalid output from model inference. ".to_string(),
        ));
    };
    let abs_max = max_audio_value.max(min_audio_value.abs());
    let audio_scale = MAX_WAV_VALUE / abs_max.max(0.01f32);
    let samples: Vec<i16> = audio_f32
        .iter()
        .map(|i| (i * audio_scale).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect::<Vec<i16>>();
    Ok(samples.into())
}

fn load_model_config(config_path: &Path) -> PiperResult<(ModelConfig, SynthesisConfig)> {
    let file = match File::open(config_path) {
        Ok(file) => file,
        Err(why) => {
            return Err(PiperError::FailedToLoadResource(format!(
                "Faild to load model config: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let model_config: ModelConfig = match serde_json::from_reader(file) {
        Ok(config) => config,
        Err(why) => {
            return Err(PiperError::FailedToLoadResource(format!(
                "Faild to parse model config from file: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let synth_config = SynthesisConfig {
        speaker: None,
        noise_scale: model_config.inference.noise_scale,
        length_scale: model_config.inference.length_scale,
        noise_w: model_config.inference.noise_w,
    };
    Ok((model_config, synth_config))
}

fn create_tashkeel_engine(
    config: &ModelConfig,
) -> PiperResult<Option<libtashkeel_base::DynamicInferenceEngine>> {
    if config.espeak.voice == "ar" {
        match libtashkeel_base::create_inference_engine(None) {
            Ok(engine) => Ok(Some(engine)),
            Err(msg) => Err(PiperError::OperationError(format!(
                "Failed to create inference engine for libtashkeel. {}",
                msg
            ))),
        }
    } else {
        Ok(None)
    }
}

fn create_inference_session(
    model_path: &Path,
    ort_env: &'static Arc<Environment>,
) -> Result<ort::Session, ort::OrtError> {
    SessionBuilder::new(ort_env)?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_model_from_file(model_path)
}

pub fn from_config_path(
    config_path: &Path,
    ort_env: &'static Arc<Environment>,
) -> PiperResult<Arc<dyn VitsVoice + Send + Sync>> {
    let (config, synth_config) = load_model_config(config_path)?;
    if config.streaming.unwrap_or_default() {
        Ok(Arc::new(VitsStreamingModel::from_config(
            config,
            synth_config,
            &config_path.with_file_name("encoder.onnx"),
            &config_path.with_file_name("decoder.onnx"),
            ort_env,
        )?))
    } else {
        let Some(onnx_filename) = config_path.file_stem() else {
            return Err(PiperError::OperationError(format!(
                "Invalid config filename format `{}`",
                config_path.display()
            )));
        };
        Ok(Arc::new(VitsModel::from_config(
            config,
            synth_config,
            &config_path.with_file_name(onnx_filename),
            ort_env,
        )?))
    }
}

#[derive(Deserialize, Default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub quality: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct ESpeakConfig {
    voice: String,
}

#[derive(Deserialize, Default, Clone)]
pub struct InferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Clone, Deserialize, Default)]
pub struct Language {
    code: String,
    #[allow(dead_code)]
    family: Option<String>,
    #[allow(dead_code)]
    region: Option<String>,
    #[allow(dead_code)]
    name_native: Option<String>,
    #[allow(dead_code)]
    name_english: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct ModelConfig {
    pub key: Option<String>,
    pub language: Option<Language>,
    pub audio: AudioConfig,
    pub num_speakers: u32,
    pub speaker_id_map: HashMap<String, i64>,
    streaming: Option<bool>,
    espeak: ESpeakConfig,
    inference: InferenceConfig,
    #[allow(dead_code)]
    num_symbols: u32,
    #[allow(dead_code)]
    phoneme_map: HashMap<i64, char>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

#[derive(Debug, Clone, Default)]
pub struct SynthesisConfig {
    pub speaker: Option<(String, i64)>,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

pub trait VitsModelCommons {
    fn get_synth_config(&self) -> &RwLock<SynthesisConfig>;
    fn get_config(&self) -> &ModelConfig;
    fn get_speaker_map(&self) -> &HashMap<i64, String>;
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine>;
    fn get_meta_ids(&self) -> (i64, i64, i64) {
        let config = self.get_config();
        let pad_id = *config.phoneme_id_map.get(&PAD).unwrap().first().unwrap();
        let bos_id = *config.phoneme_id_map.get(&BOS).unwrap().first().unwrap();
        let eos_id = *config.phoneme_id_map.get(&EOS).unwrap().first().unwrap();
        (pad_id, bos_id, eos_id)
    }
    fn language(&self) -> Option<String> {
        self.get_config()
            .language
            .as_ref()
            .map(|lang| lang.code.clone())
    }
    fn quality(&self) -> Option<String> {
        self.get_config().audio.quality.as_ref().cloned()
    }
    fn default_synthesis_config(&self) -> SynthesisConfig {
        let config = self.get_config();

        let speaker = if config.num_speakers > 0 {
            self.get_speaker_map()
                .get(&0)
                .as_ref()
                .cloned()
                .map(|name| (name.clone(), 0))
        } else {
            None
        };
        SynthesisConfig {
            speaker,
            length_scale: config.inference.length_scale,
            noise_scale: config.inference.noise_scale,
            noise_w: config.inference.noise_w,
        }
    }
    fn speakers(&self) -> PiperResult<HashMap<i64, String>> {
        Ok(self.get_speaker_map().clone())
    }
    fn get_speaker(&self) -> PiperResult<Option<String>> {
        if self.get_config().num_speakers == 0 {
            return Err(PiperError::OperationError(
                "This model is a single speaker model.".to_string(),
            ));
        }
        if let Some(ref speaker) = self.get_synth_config().read().unwrap().speaker {
            Ok(Some(speaker.0.clone()))
        } else {
            let default_speaker = match self.get_speaker_map().get(&0) {
                Some(name) => name.clone(),
                None => "Default".to_string(),
            };
            Ok(Some(default_speaker))
        }
    }
    fn set_speaker(&self, name: String) -> PiperResult<()> {
        let config = self.get_config();
        if config.num_speakers == 0 {
            return Err(PiperError::OperationError(
                "This model is a single speaker model.".to_string(),
            ));
        }
        if let Some(sid) = config.speaker_id_map.get(&name) {
            let mut synth_config = self.get_synth_config().write().unwrap();
            synth_config.speaker = Some((name, *sid));
            Ok(())
        } else {
            Err(PiperError::OperationError(format!(
                "Invalid speaker name: `{}`",
                name
            )))
        }
    }
    fn get_noise_scale(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().noise_scale)
    }
    fn set_noise_scale(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().noise_scale = value;
        Ok(())
    }
    fn get_length_scale(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().length_scale)
    }
    fn set_length_scale(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().length_scale = value;
        Ok(())
    }
    fn get_noise_w(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().noise_w)
    }
    fn set_noise_w(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().noise_w = value;
        Ok(())
    }
    fn phonemes_to_input_ids(
        &self,
        phonemes: &str,
        pad_id: i64,
        bos_id: i64,
        eos_id: i64,
    ) -> Vec<i64> {
        let config = self.get_config();
        let mut phoneme_ids: Vec<i64> = Vec::with_capacity((phonemes.len() + 1) * 2);
        phoneme_ids.push(bos_id);
        for phoneme in phonemes.chars() {
            if let Some(id) = config.phoneme_id_map.get(&phoneme) {
                phoneme_ids.push(*id.first().unwrap());
                phoneme_ids.push(pad_id);
            }
        }
        phoneme_ids.push(eos_id);
        phoneme_ids
    }
    fn do_phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        let config = self.get_config();
        let text = if config.espeak.voice == "ar" {
            let diacritized = self.diacritize_text(text)?;
            Cow::from(diacritized)
        } else {
            Cow::from(text)
        };
        let phonemes = match text_to_phonemes(&text, &config.espeak.voice, None, true, false) {
            Ok(ph) => ph,
            Err(e) => {
                return Err(PiperError::PhonemizationError(format!(
                    "Failed to phonemize given text using espeak-ng. Error: {}",
                    e
                )))
            }
        };
        Ok(phonemes.into())
    }
    fn diacritize_text(&self, text: &str) -> PiperResult<String> {
        let diacritized_text = match do_tashkeel(self.get_tashkeel_engine().unwrap(), text, None) {
            Ok(d_text) => d_text,
            Err(msg) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to diacritize text using  libtashkeel. {}",
                    msg
                )))
            }
        };
        Ok(diacritized_text)
    }
    fn get_wave_info(&self) -> PiperResult<PiperWaveInfo> {
        Ok(PiperWaveInfo {
            sample_rate: self.get_config().audio.sample_rate as usize,
            num_channels: 1usize,
            sample_width: 2usize,
        })
    }
}

pub trait VitsVoice: VitsModelCommons + PiperModel + Send + Sync {}

pub struct VitsModel {
    synth_config: RwLock<SynthesisConfig>,
    config: ModelConfig,
    speaker_map: HashMap<i64, String>,
    session: ort::Session,
    tashkeel_engine: Option<libtashkeel_base::DynamicInferenceEngine>,
}

impl VitsModel {
    pub fn new(
        config_path: PathBuf,
        onnx_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> PiperResult<Self> {
        match load_model_config(&config_path) {
            Ok((config, synth_config)) => {
                Self::from_config(config, synth_config, onnx_path, ort_env)
            }
            Err(error) => Err(error),
        }
    }
    fn from_config(
        config: ModelConfig,
        synth_config: SynthesisConfig,
        onnx_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> PiperResult<Self> {
        let session = match create_inference_session(onnx_path, ort_env) {
            Ok(session) => session,
            Err(err) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let speaker_map = reversed_mapping(&config.speaker_id_map);
        let tashkeel_engine = if config.espeak.voice == "ar" {
            match libtashkeel_base::create_inference_engine(None) {
                Ok(engine) => Some(engine),
                Err(msg) => {
                    return Err(PiperError::OperationError(format!(
                        "Failed to create inference engine for libtashkeel. {}",
                        msg
                    )))
                }
            }
        } else {
            None
        };
        Ok(Self {
            synth_config: RwLock::new(synth_config),
            config,
            speaker_map,
            session,
            tashkeel_engine,
        })
    }
    fn infer_with_values(&self, input_phonemes: Vec<i64>) -> PiperWaveResult {
        let synth_config = self.synth_config.read().unwrap();

        let input_len = input_phonemes.len();
        let phoneme_inputs =
            CowArray::from(Array2::<i64>::from_shape_vec((1, input_len), input_phonemes).unwrap())
                .into_dyn();

        let input_lengths = CowArray::from(Array1::<i64>::from_iter([input_len as i64])).into_dyn();

        let scales = Array1::<f32>::from_iter([
            synth_config.noise_scale,
            synth_config.length_scale,
            synth_config.noise_w,
        ]);
        let scales = CowArray::from(scales).into_dyn();

        let speaker_id = if self.config.num_speakers > 1 {
            let sid = match synth_config.speaker {
                Some((_, sid)) => sid,
                None => 0,
            };
            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

        let session = &self.session;
        let timer = std::time::Instant::now();
        let outputs: Vec<Value> = {
            let mut inputs = vec![
                Value::from_array(session.allocator(), &phoneme_inputs).unwrap(),
                Value::from_array(session.allocator(), &input_lengths).unwrap(),
                Value::from_array(session.allocator(), &scales).unwrap(),
            ];
            if let Some(ref sid_tensor) = speaker_id {
                inputs.push(Value::from_array(session.allocator(), sid_tensor).unwrap());
            }
            match session.run(inputs) {
                Ok(out) => out,
                Err(e) => {
                    return Err(PiperError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            }
        };
        let inference_ms = timer.elapsed().as_millis() as f32;

        let outputs: OrtOwnedTensor<f32, _> = match outputs[0].try_extract() {
            Ok(out) => out,
            Err(e) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };

        let audio_output = outputs.view();

        let samples = audio_float_to_i16(audio_output.view())?;
        Ok(PiperWaveSamples::new(
            samples,
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    pub fn get_input_output_info(&self) -> PiperResult<Vec<String>> {
        Ok(self
            .session
            .inputs
            .iter()
            .map(|i| {
                let name = i.name.clone();
                let dim: Vec<String> = i
                    .dimensions
                    .iter()
                    .map(|o| o.unwrap_or(42).to_string())
                    .collect();
                let dt = i.input_type;
                format!("#name: {}#dims: {}#type: {:?}", name, dim.join(", "), dt)
            })
            .collect())
    }
}

impl VitsModelCommons for VitsModel {
    fn get_synth_config(&self) -> &RwLock<SynthesisConfig> {
        &self.synth_config
    }
    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    fn get_speaker_map(&self) -> &HashMap<i64, String> {
        &self.speaker_map
    }
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine> {
        self.tashkeel_engine.as_ref()
    }
}

impl PiperModel for VitsModel {
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        self.do_phonemize_text(text)
    }

    fn speak_batch(&self, phoneme_batches: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phoneme_batches = Vec::from_iter(
            phoneme_batches
                .into_iter()
                .map(|phonemes| self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id)),
        );
        let mut retval = Vec::new();
        for phonemes in phoneme_batches.into_iter() {
            retval.push(self.infer_with_values(phonemes)?);
        }
        Ok(retval)
    }

    fn speak_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        self.infer_with_values(phonemes)
    }
    fn wave_info(&self) -> PiperResult<PiperWaveInfo> {
        self.get_wave_info()
    }
}

impl VitsVoice for VitsModel {}

pub struct VitsStreamingModel {
    synth_config: RwLock<SynthesisConfig>,
    config: ModelConfig,
    speaker_map: HashMap<i64, String>,
    encoder_model: ort::Session,
    decoder_model: Arc<ort::Session>,
    tashkeel_engine: Option<libtashkeel_base::DynamicInferenceEngine>,
}

impl VitsStreamingModel {
    fn from_config(
        config: ModelConfig,
        synth_config: SynthesisConfig,
        encoder_path: &Path,
        decoder_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> PiperResult<Self> {
        let encoder_model = match create_inference_session(encoder_path, ort_env) {
            Ok(model) => model,
            Err(err) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let decoder_model = match create_inference_session(decoder_path, ort_env) {
            Ok(model) => Arc::new(model),
            Err(err) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let speaker_map = reversed_mapping(&config.speaker_id_map);
        let tashkeel_engine = create_tashkeel_engine(&config)?;
        Ok(Self {
            synth_config: RwLock::new(synth_config),
            config,
            speaker_map,
            encoder_model,
            decoder_model,
            tashkeel_engine,
        })
    }

    fn infer_with_values(&self, input_phonemes: Vec<i64>) -> PiperWaveResult {
        let timer = std::time::Instant::now();
        let encoder_output = self.infer_encoder(input_phonemes)?;
        let audio = self.infer_decoder(encoder_output)?;
        let inference_ms = timer.elapsed().as_millis() as f32;
        Ok(PiperWaveSamples::new(
            audio,
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    fn infer_encoder(&self, input_phonemes: Vec<i64>) -> PiperResult<EncoderOutputs> {
        let synth_config = self.synth_config.read().unwrap();

        let input_len = input_phonemes.len();
        let phoneme_inputs =
            CowArray::from(Array2::<i64>::from_shape_vec((1, input_len), input_phonemes).unwrap())
                .into_dyn();

        let input_lengths = CowArray::from(Array1::<i64>::from_iter([input_len as i64])).into_dyn();

        let scales = Array1::<f32>::from_iter([
            synth_config.noise_scale,
            synth_config.length_scale,
            synth_config.noise_w,
        ]);
        let scales = CowArray::from(scales).into_dyn();

        let speaker_id = if self.config.num_speakers > 1 {
            let sid = match synth_config.speaker {
                Some((_, sid)) => sid,
                None => 0,
            };
            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

        let session = &self.encoder_model;
        let ort_values: Vec<Value> = {
            let mut inputs = vec![
                Value::from_array(session.allocator(), &phoneme_inputs).unwrap(),
                Value::from_array(session.allocator(), &input_lengths).unwrap(),
                Value::from_array(session.allocator(), &scales).unwrap(),
            ];
            if let Some(ref sid_tensor) = speaker_id {
                inputs.push(Value::from_array(session.allocator(), sid_tensor).unwrap());
            }
            match session.run(inputs) {
                Ok(out) => out,
                Err(e) => {
                    return Err(PiperError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            }
        };
        EncoderOutputs::new(ManuallyDrop::new(ort_values))
    }
    fn infer_decoder(&self, encoder_outputs: EncoderOutputs) -> PiperResult<RawWaveSamples> {
        encoder_outputs.infer_decoder(self.decoder_model.as_ref())
    }
}

impl VitsModelCommons for VitsStreamingModel {
    fn get_synth_config(&self) -> &RwLock<SynthesisConfig> {
        &self.synth_config
    }
    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    fn get_speaker_map(&self) -> &HashMap<i64, String> {
        &self.speaker_map
    }
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine> {
        self.tashkeel_engine.as_ref()
    }
}

impl PiperModel for VitsStreamingModel {
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        self.do_phonemize_text(text)
    }

    fn speak_batch(&self, phoneme_batches: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phoneme_batches = Vec::from_iter(
            phoneme_batches
                .into_iter()
                .map(|phonemes| self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id)),
        );
        let mut retval = Vec::new();
        for phonemes in phoneme_batches.into_iter() {
            retval.push(self.infer_with_values(phonemes)?);
        }
        Ok(retval)
    }
    fn speak_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        self.infer_with_values(phonemes)
    }
    fn wave_info(&self) -> PiperResult<PiperWaveInfo> {
        self.get_wave_info()
    }
    fn stream_synthesis<'a>(
        &'a self,
        phonemes: String,
        chunk_size: u32,
        chunk_padding: u32,
    ) -> PiperResult<Box<dyn Iterator<Item = PiperResult<RawWaveSamples>> + 'a>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        let encoder_outputs = self.infer_encoder(phonemes)?;
        let streamer = Box::new(SpeechStreamer {
            decoder_model: Arc::clone(&self.decoder_model),
            encoder_outputs,
            chunk_size,
            chunk_padding
        });
        Ok(streamer)
    }
}

impl VitsVoice for VitsStreamingModel {}

struct EncoderOutputs<'a> {
    values: OnceCell<ManuallyDrop<Vec<Value<'static>>>>,
    z: OrtOwnedTensor<'a, f32, Dim<IxDynImpl>>,
    y_mask: OrtOwnedTensor<'a, f32, Dim<IxDynImpl>>,
    g: Array<f32, Dim<IxDynImpl>>,
}

impl<'a> EncoderOutputs<'a> {
    fn new(values: ManuallyDrop<Vec<Value<'static>>>) -> PiperResult<Self> {
        let z: OrtOwnedTensor<f32, _> = match values[0].try_extract() {
            Ok(out) => out,
            Err(e) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };
        let y_mask: OrtOwnedTensor<f32, _> = match values[1].try_extract() {
            Ok(out) => out,
            Err(e) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };
        let g = if values.len() == 3 {
            let g_t: OrtOwnedTensor<f32, _> = match values[2].try_extract() {
                Ok(out) => out,
                Err(e) => {
                    return Err(PiperError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            };
            g_t.view().clone().into_owned()
        } else {
            Array1::<f32>::from_iter([]).into_dyn()
        };
        Ok(Self {
            values: OnceCell::with_value(values),
            z,
            y_mask,
            g,
        })
    }
    fn infer_decoder(&self, session: &ort::Session) -> PiperResult<RawWaveSamples> {
        let outputs: Vec<Value> = {
            let z_view = self.z.view();
            let y_mask_view = self.y_mask.view();
            let z_input = CowArray::from(z_view.view());
            let y_mask_input = CowArray::from(y_mask_view.view());
            let g_input = CowArray::from(self.g.view());
            let mut inputs = vec![
                Value::from_array(session.allocator(), &z_input).unwrap(),
                Value::from_array(session.allocator(), &y_mask_input).unwrap(),
            ];
            if !g_input.is_empty() {
                inputs.push(Value::from_array(session.allocator(), &g_input).unwrap())
            }
            match session.run(inputs) {
                Ok(out) => out,
                Err(e) => {
                    return Err(PiperError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            }
        };
        match outputs[0].try_extract() {
            Ok(out) => audio_float_to_i16(out.view().view()),
            Err(e) => Err(PiperError::OperationError(format!(
                "Failed to run model inference. Error: {}",
                e
            ))),
        }
    }
}

impl<'a> Drop for EncoderOutputs<'a> {
    fn drop(&mut self) {
        if let Some(values) = self.values.take() {
            ManuallyDrop::into_inner(values);
        }
    }
}

struct SpeechStreamer<'a> {
    decoder_model: Arc<ort::Session>,
    encoder_outputs: EncoderOutputs<'a>,
    chunk_size: u32,
    chunk_padding: u32
}

impl<'a> Iterator for SpeechStreamer<'a> {
    type Item = PiperResult<RawWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}