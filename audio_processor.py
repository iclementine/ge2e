from scipy.ndimage.morphology import binary_dilation
from config import get_cfg_defaults
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad = None

int16_max = (2 ** 15) -1

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    # dBFS: Decibels relative to full scale 
    # See https://en.wikipedia.org/wiki/DBFS for more details
    # for 16Bit PCM audio, minimal level is -96dB
    # compute the mean dBFS and adjust to target dBFS, with by increasing or decreasing
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav, 
                       vad_window_length: int, 
                       vad_moving_average_width: int, 
                       vad_max_silence_length: int, 
                       sampling_rate: int):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


class SpeakerVerificationPreprocessor(object):
    def __init__(self, 
                sampling_rate,
                audio_norm_target_dBFS,
                vad_window_length,
                vad_moving_average_width,
                vad_max_silence_length,
                mel_window_length,
                mel_window_step,
                n_mels):
        self.sampling_rate = sampling_rate
        self.audio_norm_target_dBFS = audio_norm_target_dBFS
        
        self.vad_window_length = vad_window_length
        self.vad_moving_average_width = vad_moving_average_width
        self.vad_max_silence_length = vad_max_silence_length

        self.n_fft = int(mel_window_length * sampling_rate / 1000)
        self.hop_length = int(mel_window_step * sampling_rate / 1000)
        self.n_mels = n_mels

    def preprocess_wav(self, fpath_or_wav, source_sr=None):
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
        else:
            wav = fpath_or_wav
        
        # Resample if numpy.array is passed and sr does not match
        if source_sr is not None and source_sr != self.sampling_rate:
            wav = librosa.resample(wav, source_sr, self.sampling_rate)
        
        # normalize of
        wav = normalize_volume(wav, self.audio_norm_target_dBFS, increase_only=True)

        # trim long silence
        if webrtcvad:
            wav = trim_long_silences(
                wav, 
                self.vad_window_length, 
                self.vad_moving_average_width, 
                self.vad_max_silence_length, 
                self.sampling_rate)
        return wav

    def melspectrogram(self, wav):
        mel = librosa.feature.melspectrogram(
            wav,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels)
        mel = mel.astype(np.float32).T
        return mel

