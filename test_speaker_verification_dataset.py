from speaker_verification_dataset import SpeakerVerificationDataset, SpeakerVerificationSampler, RandomClip
from pprint import pprint
from parakeet.data.dataset import TransformDataset
from paddle.io import DataLoader

dataset = SpeakerVerificationDataset("/home/chenfeiyu/datasets/SV2TTS/encoder")
sampler = SpeakerVerificationSampler(dataset, 64, 10)
import pdb; pdb.set_trace()
partial_dataset = TransformDataset(dataset, RandomClip(160))

dataloader = DataLoader(partial_dataset, batch_sampler=sampler)

for i, batch in enumerate(dataloader):
    print(batch[0].shape)
    if i == 10:
        break
