import argparse
from collections import Counter, defaultdict
from itertools import chain

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import QuickThoughtsModel

MAX_LEN = 60
WORD_EMBED = 128
THOUGHT_SIZE = 256


class Corpus(Dataset):
    def __init__(self, max_length=MAX_LEN, unk_threshold=5):
        self.word_index = defaultdict(lambda: len(self.word_index))
        self.max_length = max_length
        self.unk_threshold = unk_threshold
        self.dataset = list()

    @staticmethod
    def read_corpus(corpus_file, tag_file):
        self = Corpus()

        # self.word_index['EOS']  # 0
        # self.word_index['UNK']  # 1

        with open(corpus_file, 'r', encoding='utf8') as fc, open(tag_file, 'r', encoding='utf8') as ft:
            sentences = [line.split() for line in fc if line]
            tags = [line.split() for line in ft if line]
            counter = Counter(chain.from_iterable(sentences))
            dataset = []

            for sent, tag in zip(sentences, tags):
                word_indices = list()
                tag_indices = list()
                for word, word_tag in zip(sent[:self.max_length], tag[:self.max_length]):
                    if counter[word] <= self.unk_threshold:
                        word = 'UNK'
                    word_indices.append(self.word_index[word])
                    tag_indices.append(1) if word_tag in ['1', '2', '0'] else tag_indices.append(0)

                if len(word_indices) < self.max_length:
                    word_indices += ([self.word_index['EOS']] * (self.max_length - len(word_indices)))
                    tag_indices += ([0] * (self.max_length - len(tag_indices)))
                dataset.append([word_indices, tag_indices])
                # 183768, 2, 80

        self.index_word = {v: k for k, v in self.word_index.items()}
        self.dataset = torch.LongTensor(dataset)

        return self

    def vocab_count(self):
        return len(self.word_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train(model, dataset, args, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False)
    for epoch in range(1, args.epoch + 1):
        print('Epoch %d' % epoch)
        train_loss_quick = 0
        train_loss_ind = 0
        with tqdm(total=len(train_loader)) as bar:
            for batch_idx, batch in enumerate(train_loader):
                loss_quick, loss_ind = model(batch.to(device))
                optimizer.zero_grad()

                loss = loss_quick + loss_ind
                loss.backward()

                optimizer.step()
                train_loss_quick += loss_quick.item()
                train_loss_ind += loss_ind.item()
                bar.set_postfix_str('loss_quick:%.4f|avg_loss:%.4f loss_ind:%.4f|avg_loss:%.4f' %
                                    (loss_quick.item(), train_loss_quick / (batch_idx + 1),
                                     loss_ind.item(), train_loss_ind / (batch_idx + 1)))
                bar.update()
    torch.save(train_loader.dataset.index_word, args.dict)
    torch.save(model.state_dict(), args.save)


def parse_args():
    epoch = 10
    batch_size = 256
    context_size = 1
    dropout = 0.
    bidirectional = False
    pretrained_weight = None

    parser = argparse.ArgumentParser(description='Quick-Thought Vectors')

    parser.add_argument('--train', type=str, default='data/classify_data.seg',
                        help='source corpus file')
    parser.add_argument('--save', type=str, default='quick_thought_main_all.bin',
                        help='path to save the final model')
    parser.add_argument('--tags', type=str, default='data/classify.tag',
                        help='tag file for source corpus file')
    parser.add_argument('--dict', type=str, default='data/dict.bin',
                        help='path to save dictionary')
    parser.add_argument('--epoch', '-e', default=epoch, metavar='N', type=int,
                        help=f'number of training epochs (default: {epoch})')
    parser.add_argument('--batch_size', '-b', default=batch_size,
                        metavar='N', type=int,
                        help=f'minibatch size for training (default: {batch_size})')
    parser.add_argument('--wembed', '-w', default=WORD_EMBED,
                        metavar='N', type=int,
                        help=f'the dimension of word embedding (default: {WORD_EMBED})')
    parser.add_argument('--sembed', '-s', default=THOUGHT_SIZE,
                        metavar='N', type=int,
                        help=f'the dimension of sentence embedding (default: {THOUGHT_SIZE})')
    parser.add_argument('--context', '-c', default=context_size,
                        metavar='N', type=int,
                        help=f'predict previous and next N sentences (default: {context_size})')
    parser.add_argument('--dropout', '-d', default=dropout,
                        metavar='N', type=int,
                        help=f'dropout rate (default: {dropout})')
    parser.add_argument('--bidirectional', default=bidirectional,
                        type=bool, choices=[True, False],
                        help=f'use bi-directional model (default: {bidirectional})')
    parser.add_argument('--pretrained', default=pretrained_weight, type=str,
                        help='pre-trained word embeddings file')
    parser.add_argument('--seed', type=int, default='1234', help='random seed')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    dataset = Corpus.read_corpus(args.train, args.tags)
    vocab_size = dataset.vocab_count()

    model = QuickThoughtsModel(vocab_size, args.wembed, args.sembed,
                               args.context, dropout=args.dropout,
                               bidirectional=args.bidirectional,
                               pretrained_weight=args.pretrained,
                               device=device)
    model.to(device)
    print(model)

    train(model, dataset, args, device)


if __name__ == '__main__':
    main()
