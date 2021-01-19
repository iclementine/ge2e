import random
import numpy as np
import paddle
from pathlib import Path
from random_cycle import random_cycle

class SpeakerVerificationDataset(paddle.io.Dataset):
    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root).resolve()
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        
        speaker_utterances = {
            speaker_dir: [f for f in speaker_dir.glob("*.npy")]
            for speaker_dir in speaker_dirs
        }

        self.speaker_dirs = speaker_dirs
        self.speaker_utterances = speaker_utterances

    def num_speakers(self):
        return len(self.speaker_dirs)
    
    def num_utterances(self):
        return np.sum(len(utterances) \
            for speaker, utterances in self.speaker_utterances.items())
    
    def utterances_counts(self):
        return self.speaker_utterances

    def get_example_by_index(self, speaker_index, utterance_index):
        speaker_dir = self.speaker_dirs[speaker_index]
        fpath = self.speaker_utterances[speaker_dir][utterance_index]
        return self[fpath]

    def __getitem__(self, fpath):
        return np.load(fpath)

    def __len__(self):
        return int(self.num_utterances())

class RandomClip(object):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, spec):
        T = spec.shape[0]
        start = random.randint(0, T - self.frames)
        return spec[start: start+ self.frames, :]

class SpeakerVerificationSampler(paddle.io.BatchSampler):
    def __init__(self, dataset: SpeakerVerificationDataset, speakers_per_batch, utterances_per_speaker):
        self._speakers = list(dataset.speaker_dirs)
        self._speaker_utterances = dataset.speaker_utterances

        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

    def __iter__(self):
        speaker_generator = iter(random_cycle(self._speakers))
        speaker_utterances_generator = {s: iter(random_cycle(us)) for s, us in self._speaker_utterances.items()}

        while True:
            speakers = []
            for _ in range(self.speakers_per_batch):
                speakers.append(next(speaker_generator))

            utterances = []
            for s in speakers:
                us = speaker_utterances_generator[s]
                for _ in range(self.utterances_per_speaker):
                    utterances.append(next(us))
            yield utterances
        

if __name__ == "__main__":
    mydataset = SpeakerVerificationDataset(Path("/home/chenfeiyu/datasets/SV2TTS/encoder"))
    print(mydataset.get_example_by_index(0, 10))