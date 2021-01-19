import paddle
import random
import numpy as np

class MyKeyDataset(paddle.io.Dataset):
    def __init__(self, datasource: dict):
        self._data = datasource
        self._len = len(datasource)
        self._keys = list(datasource.keys())

    def __getitem__(self, key):
        return self._data[key]

class MySampler(paddle.io.Sampler):
    def __init__(self, dataset, N, M):
        self._keys = dataset._keys
        self._speakers = {
            str(i): [key for key in self._keys if key.endswith(str(i))]
            for i in range(10)
        }
        self.N = N
        self.M = M
    
    def __iter__(self):
        while True:
            speakers = list(self._speakers.keys())
            speaker_ids = np.random.choice(speakers, self.N, replace=False)
            utterance_ids = []
            for s in speaker_ids:
                utterance_ids.append(np.random.choice(self._speakers[s], self.M))
            utterance_ids = np.concatenate(utterance_ids).tolist()
            yield utterance_ids



my_dataset = MyKeyDataset({str(i): np.array([i, i**2, i**3]) for i in range(100)})
print(my_dataset)
print(my_dataset['12'])

my_sampler = MySampler(my_dataset, 5, 4)
iterator = iter(my_sampler)

print(next(iterator))
loader = paddle.io.DataLoader(my_dataset, batch_sampler=my_sampler)
data_iterator = iter(loader)
print(next(data_iterator))

