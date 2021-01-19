import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

class SpeakerEncoder(nn.Module):
    # 实在是有点丑陋， 虽然这是一个 model parallel 的模型，但是把设备作为参数还是有点丑陋啊
    # 而且所有的形状相关的东西竟然都不在这里面， 而是弄成了 hparams
    def __init__(self, mel_dim, num_layers, hidden_size, output_size):
        super().__init__()
        
        # Network defition
        self.lstm = nn.LSTM(input_size=mel_dim,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=output_size)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([1.]))
        self.similarity_bias = nn.Parameter(torch.tensor([0.]))
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker, embed_dim = embeds.shape
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1)
        centroids_incl_norm = torch.norm(centroids_incl, dim=-1, keepdim=True)
        normalized_centroids_incl = centroids_incl / centroids_incl_norm

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl_norm = torch.norm(centroids_excl, dim=2, keepdim=True)
        normalized_centroids_excl = centroids_excl / centroids_excl_norm

        p1 = torch.matmul(embeds.reshape(-1, embed_dim), normalized_centroids_incl.transpose(1, 0)) # (NM, N)
        # print(p1.shape)
        p2 = torch.bmm(embeds.reshape(-1, 1, embed_dim), normalized_centroids_excl.reshape(-1, embed_dim, 1)).squeeze(-1) # （NM, 1)
        # print(p2.shape)
        # __DEBUG__.append(embeds); embeds.retain_grad()
        index = torch.repeat_interleave(torch.arange(speakers_per_batch), utterances_per_speaker).unsqueeze(-1).to(p1.device)
        p = torch.scatter(p1, 1, index, p2)
        
        p = p * self.similarity_weight + self.similarity_bias # neg
        return p, p1, p2

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        target = torch.repeat_interleave(torch.arange(speakers_per_batch), utterances_per_speaker).to(sim_matrix.device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            ground_truth = target.data.cpu().numpy()
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer