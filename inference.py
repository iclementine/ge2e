from model import SpeakerEncoder
from audio_processor import SpeakerVerificationPreprocessor
from matplotlib import cm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import get_cfg_defaults
import argparse
import paddle
import tqdm

_model = None # type: SpeakerEncoder


# 加载 speaker encoder 模型
def load_model(config, weights_fpath: Path):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model
    _model = SpeakerEncoder(
            config.data.n_mels, 
            config.model.num_layers,
            config.model.hidden_size,
            config.model.embedding_size)
    model_state_dict = paddle.load(weights_fpath + ".pdparams")
    _model.set_state_dict(model_state_dict)
    _model.eval()
    print(f"Loaded encoder {weights_fpath}")
    
    
def is_loaded():
    return _model is not None


# 提取 speaker embed(from batch of partials)
def embed_frames_batch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram.
    
    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = paddle.to_tensor(frames_batch)
    with paddle.no_grad():
        embed = _model(frames).numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames, hop_length, 
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    # samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / hop_length))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * hop_length
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices


def embed_utterance(processor, wav, partial_utterance_n_frames, using_partials=True, return_partials=False):
    """
    Computes an embedding for a single utterance.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = processor.melspectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), partial_utterance_n_frames, processor.hop_length)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    frames = processor.melspectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    # TODO: batch multiple utterances' partial specs together
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplemented()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)
    
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)

def main(config, args):
    paddle.set_device(args.device)
    load_model(config, args.checkpoint_path)

    c = config.data
    processor = SpeakerVerificationPreprocessor(
        c.sampling_rate, c.audio_norm_target_dBFS, c.vad_window_length,
        c.vad_moving_average_width, c.vad_max_silence_length,
        c.mel_window_length, c.mel_window_step, c.n_mels)

    input_dir = Path(args.input).expanduser()
    ifpaths = list(input_dir.rglob("*.wav"))
    print(f"{len(ifpaths)} utterances in total")
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ifpath in tqdm.tqdm(ifpaths):
        rel_path = ifpath.relative_to(input_dir)
        ofpath = (output_dir / rel_path).with_suffix("")
        ofpath.parent.mkdir(parents=True, exist_ok=True)
        wav = processor.preprocess_wav(ifpath)
        embed = embed_utterance(processor, wav, c.partials_n_frames, using_partials=True, return_partials=False)
        np.save(ofpath, embed)





if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = argparse.ArgumentParser(description="compute utterance embed.")
    parser.add_argument("--config", metavar="FILE", help="path of the config file to overwrite to default config with.")
    parser.add_argument("--input", type=str, help="path of the audio_file")
    parser.add_argument("--output", metavar="OUTPUT_DIR", help="path to save checkpoint and logs.")

    # load from saved checkpoint
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load")

    # running
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], help="device type to use, cpu and gpu are supported.")
    parser.add_argument("--nprocs", type=int, default=1, help="number of parallel processes to use.")

    # overwrite extra config and default config
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="options to overwrite --config file and the default config, passing in KEY VALUE pairs")

    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
