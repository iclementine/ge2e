from yacs.config import CfgNode

_C = CfgNode()


data_config = _C.data = CfgNode()
data_config.n_mel = 40
data_config.partial_frames = 160


model_config = _C.model = CfgNode()
model_config.num_layers = 3
model_config.hidden_size = 256
model_config.embedding_size = 256 # output size

training_config = _C.training = CfgNode()
training_config.learning_rate_init = 1e-4
training_config.speakers_per_batch = 64
training_config.utterances_per_speaker = 10
training_config.max_iteration = 1560000
training_config.save_interval = 10000
training_config.valid_interval = 10000


def get_cfg_defaults():
    return _C.clone()
