from parakeet.training import ExperimentBase, default_argument_parser
from config import get_cfg_defaults
from paddle import distributed as dist
import time
from paddle.nn.clip import ClipGradByGlobalNorm

from parakeet.data.dataset import TransformDataset
from model import SpeakerEncoder
from speaker_verification_dataset import SpeakerVerificationDataset, SpeakerVerificationSampler, RandomClip
from paddle.optimizer import Adam
from paddle import DataParallel
from paddle.io import DataLoader

class Ge2eExperiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model = SpeakerEncoder(
            config.data.n_mel, 
            config.model.num_layers,
            config.model.hidden_size,
            config.model.embedding_size)
        optimizer = Adam(
            config.training.learning_rate_init, 
            parameters=model.parameters(),
            grad_clip=ClipGradByGlobalNorm(3))
        self.model = DataParallel(model) if self.parallel else model
        self.model_core = model
        self.optimizer = optimizer

    def setup_dataloader(self):
        config = self.config
        raw_dataset = SpeakerVerificationDataset(self.args.data)
        sampler = SpeakerVerificationSampler(
            raw_dataset, 
            config.training.speakers_per_batch, 
            config.training.utterances_per_speaker)
        train_dataset = TransformDataset(raw_dataset, RandomClip(config.data.partial_frames))
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=16)
        self.train_dataset = train_dataset
        self.train_loader = train_loader

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.optimizer.clear_grad()
        self.model.train()
        specs, = batch
        embeds = self.model(specs)
        N = self.config.training.speakers_per_batch
        M = self.config.training.utterances_per_speaker
        embeds = embeds.reshape([N, M, -1])
        loss, eer = self.model_core.loss(embeds)
        loss.backward()
        self.model_core.do_gradient_ops()
        self.optimizer.step()
        iteration_time = time.time() - start

        # logging
        loss_value = float(loss)
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                  iteration_time)
        msg += 'loss: {:>.6f} err: {:>.6f}'.format(loss_value, eer)
        self.logger.info(msg)

        if dist.get_rank() == 0:
            self.visualizer.add_scalar("train/loss", loss_value, self.iteration)
            self.visualizer.add_scalar("train/eer", eer, self.iteration)
            self.visualizer.add_scalar("param/w", float(self.model.similarity_weight), self.iteration)
            self.visualizer.add_scalar("param/b", float(self.model.similarity_bias), self.iteration)


    def valid(self):
        pass


def main_sp(config, args):
    exp = Ge2eExperiment(config, args)
    exp.setup()
    exp.run()

def main(config, args):
    if args.nprocs > 1 and args.device == "gpu":
        dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    else:
        main_sp(config, args)

if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)