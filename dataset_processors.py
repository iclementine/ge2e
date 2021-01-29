import numpy as np
from pathlib import Path
import multiprocessing as mp
from audio_processor import SpeakerVerificationPreprocessor
from tqdm import tqdm
from functools import partial

def _process_utterance(path_pair,
                       processor: SpeakerVerificationPreprocessor,
                       partials_n_frames: int):
    # Load and preprocess the waveform
    input_path, output_path = path_pair
    wav = processor.preprocess_wav(input_path)
    if len(wav) == 0:
        return
    
    # Create the mel spectrogram, discard those that are too short
    frames = processor.melspectrogram(wav)
    if len(frames) < partials_n_frames:
        return
    
    np.save(output_path, frames)


def _process_speaker(speaker_dir, processor, datasets_root, output_dir, extension, partials_n_frames, skip_existing=False):
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
    speaker_output_dir = output_dir / speaker_name
    speaker_output_dir.mkdir(exist_ok=True)
    
    # load exsiting file set
    sources_fpath = speaker_output_dir / "_sources.txt"
    if sources_fpath.exists():
        try:
            with sources_fpath.open("rt") as sources_file:
                existing_names = {line.split(",")[0] for line in sources_file}
        except:
            existing_names = {}
    else:
        existing_names = {}
    
    sources_file = sources_fpath.open("at" if skip_existing else "wt")
    for in_fpath in speaker_dir.glob(f"**/*.{extension}"):
        out_name = "_".join(in_fpath.relative_to(speaker_dir).parts)
        out_name = out_name.replace(f".{extension}", ".npy")
        if skip_existing and out_name in existing_names:
            continue
        out_fpath = speaker_output_dir / out_name
        _process_utterance((in_fpath, out_fpath), processor, partials_n_frames)
        sources_file.write(f"{out_name},{in_fpath}\n")
    
    sources_file.close()


def _process_dataset(processor, datasets_root, speaker_dirs, dataset_name, output_dir, extension, partials_n_frames, skip_existing=False):
    
    # TODO: filter speakers in VoxCeleb1
    print(f"{dataset_name}: Preprocessing data for {len(speaker_dirs)}.")

    _func = partial(_process_speaker, processor=processor, datasets_root=datasets_root, output_dir=output_dir, extension=extension, partials_n_frames=partials_n_frames, skip_existing=skip_existing)

    with mp.Pool(16) as pool:
        list(tqdm(pool.imap(_func, speaker_dirs), dataset_name, len(speaker_dirs), unit="speakers"))
    print(f"Done preprocessing {dataset_name}.")


def process_librispeech(processor, datasets_root, output_dir, partials_n_frames, skip_existing=False):
    dataset_name = "LibriSpeech/train-other-500"
    dataset_root = datasets_root / dataset_name
    speaker_dirs = list(dataset_root.glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name, output_dir, "flac", partials_n_frames, skip_existing)


def process_voxceleb1(processor, datasets_root, output_dir, partials_n_frames, skip_existing=False):
    dataset_name = "VoxCeleb1"
    dataset_root = datasets_root / dataset_name
    
    anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
    with dataset_root.joinpath("vox1_meta.csv").open("rt") as metafile:
        metadata = [line.strip().split("\t") for line in metafile][1:]
    
    # speaker id -> nationality
    nationalities = {line[0]: line[3] for line in metadata if line[-1] == "dev"}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                        nationality.lower() in anglophone_nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
          (len(keep_speaker_ids), len(nationalities)))

    speaker_dirs = list((dataset_root / "wav").glob("*"))
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                    speaker_dir.name in keep_speaker_ids]
    # TODO: filter ansa
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name, output_dir, "wav", partials_n_frames, skip_existing)


def process_voxceleb2(processor, datasets_root, output_dir, partials_n_frames, skip_existing=False):
    dataset_name = "VoxCeleb2"
    dataset_root = datasets_root / dataset_name
    # There is no nationality if meta data for VoxCeleb2
    speaker_dirs = list((dataset_root / "wav").glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name, output_dir, "wav", partials_n_frames, skip_existing)
