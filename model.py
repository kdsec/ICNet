import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickThoughtsModel(nn.Module):
    def __init__(self, vocab_size, embed_size, thought_size, context_size,
                 dropout=0.3, bidirectional=False, pretrained_weight=None,
                 device=torch.device('cuda')):
        super().__init__()
        print('Init quick thought model...')
        self.context_size = context_size
        self.encoder = Encoder(vocab_size, embed_size, thought_size, dropout,
                               bidirectional, pretrained_weight)
        self.s2w = nn.Linear(thought_size, embed_size)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, input_data):
        sentences = input_data[:, 0, :].squeeze()
        tags = input_data[:, 1, :].squeeze()

        word_embeddings, thought_vectors = self.encoder(sentences)
        batch_size = len(thought_vectors)

        scores = torch.matmul(thought_vectors, torch.t(thought_vectors))
        scores[torch.eye(batch_size).byte()] = 0

        target_np = np.zeros(scores.size(), dtype=np.int64)
        for i in range(1, self.context_size + 1):
            # the i-th previous and next sentence
            target_np += np.eye(batch_size, k=-i, dtype=np.int64)
            target_np += np.eye(batch_size, k=i, dtype=np.int64)

        # normalize target matrix by row
        target_np_sum = np.sum(target_np, axis=1, keepdims=True)
        target_np = target_np / target_np_sum
        if self.device == torch.device('cpu'):
            target = torch.from_numpy(target_np).type(torch.LongTensor)
        else:
            target = torch.from_numpy(target_np).type(torch.cuda.LongTensor)

        score = torch.cat((1 - scores.view(1, -1), scores.view(1, -1)))  # 2, 65536
        target = target.view(1, -1).view(-1)  # 65536

        loss_quick = self.criterion(torch.t(score), target)
        loss_ind = self.cal_indicator(word_embeddings, thought_vectors, tags)

        return loss_quick, loss_ind

    def cal_indicator(self, word_embeddings, thought_vectors, tags):
        softmax = torch.nn.Softmax(dim=1)
        thought_vectors = self.s2w(thought_vectors).unsqueeze(1).expand_as(word_embeddings).unsqueeze(2)
        # 256, 30, 1, 100 batch_size, sentence_len, 1, word_dim

        word_embeddings = word_embeddings.unsqueeze(3)
        # 256, 30, 100, 1 batch_size, sentence, word_dim, 1

        scores = softmax(torch.matmul(thought_vectors, word_embeddings).squeeze())
        # 256, 30 batch_size, sentence_len

        ind_mul = scores * tags.float()
        ind_sum = torch.sum(ind_mul, dim=1)
        ind_sum = torch.masked_select(ind_sum, (torch.sum(tags, dim=1) > 0))

        loss_ind = F.relu(0.4-torch.mean(ind_sum))
        return loss_ind


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, thought_size, dropout,
                 bidirectional, pretrained_weight):
        super().__init__()

        self.thought_size = thought_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

        if pretrained_weight:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weight, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(embed_size, thought_size, dropout=dropout, bidirectional=bidirectional)
        self._init_weights(pretrained_weight)

    def _init_weights(self, pretrained_weight):
        if not pretrained_weight:
            self.embedding.weight.data.uniform_(-0.1, 0.1)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sentences):
        word_embeddings = self.embedding(sentences)  # 256, 30, 100  batch_size, sentence_len, word_dim
        _, hidden = self.gru(word_embeddings.transpose(0, 1))

        if self.bidirectional:
            thought_vectors = torch.cat((hidden[0], hidden[1]), 1)
        else:
            thought_vectors = hidden[0]  # 256, 2400  batch_size, sentence_dim

        return word_embeddings, thought_vectors
