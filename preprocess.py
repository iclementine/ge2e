import argparse
from pathlib import Path
from config import get_cfg_defaults
from audio_processor import SpeakerVerificationPreprocessor
from dataset_processors import process_librispeech, process_voxceleb1, process_voxceleb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess dataset for speaker verification task")
    parser.add_argument(
        "--datasets_root",
        type=Path,
        help=
        "Path to the directory containing your LibriSpeech, LibriTTS and VoxCeleb datasets."
    )
    parser.add_argument("--output_dir",
                        type=Path,
                        help="Path to save processed dataset.")
    parser.add_argument(
        "--dataset_names",
        type=str,
        default="librispeech_other,voxceleb1,voxceleb2",
        help=
        "comma-separated list of names of the datasets you want to preprocess. only "
        "the train set of these datastes will be used. Possible names: librispeech_other, "
        "voxceleb1, voxceleb2.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=
        "Whether to skip ouput files with the same name. Useful if this script was interrupted."
    )
    parser.add_argument(
        "--no_trim",
        action="store_true",
        help="Preprocess audio without trimming silences (not recommended).")

    args = parser.parse_args()

    if not args.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError(
                "Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    args.datasets = [item.strip() for item in args.dataset_names.split(",")]
    if not hasattr(args, "output_dir"):
        args.output_dir = args.dataset_root / "SV2TTS" / "encoder"
    assert args.datasets_root.exists()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    config = get_cfg_defaults()
    # TODO: nice print
    print(args)

    c = config.data
    processor = SpeakerVerificationPreprocessor(
        c.sampling_rate, c.audio_norm_target_dBFS, c.vad_window_length,
        c.vad_moving_average_width, c.vad_max_silence_length,
        c.mel_window_length, c.mel_window_step, c.n_mels)

    preprocess_func = {
        "librispeech_other": process_librispeech,
        "voxceleb1": process_voxceleb1,
        "voxceleb2": process_voxceleb2,
    }

    for dataset in args.datasets:
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](processor, args.datasets_root, args.output_dir, c.partials_n_frames, args.skip_existing)


