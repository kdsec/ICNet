import json
import argparse
import json

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dnn import DNNNet

USE_CUDA = True
BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 60
WORD_EMBED = 128
THOUGHT_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")


# torch.manual_seed(1357)


class DataSet:
    def __init__(self, X, y):
        self.X = []
        for line, _ in X:
            if len(line) < MAX_LEN:
                line += [0] * (MAX_LEN - len(line))
            line = line[:MAX_LEN]
            self.X.append(torch.LongTensor(line).to(device))
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        target = [0, 0]
        target[self.y[index]] = 1
        return self.X[index], torch.FloatTensor(np.asarray(target)).to(device)


def train():
    # if not os.path.exists('./classifier'):
    #     os.mkdir('./classifier')

    print("Load data...")
    dataset = json.load(open('data/classify.tag.json', 'r'))
    train_dataset = DataSet(dataset['X_train'], dataset['y_train'])
    test_dataset = DataSet(dataset['X_test'], dataset['y_test'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Build model.
    dnn_model = DNNNet(thought_size=THOUGHT_SIZE)
    dnn_model.to(device)
    print(dnn_model)

    optimizer = optim.Adam(dnn_model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        dnn_model.train()
        train_loss = 0
        total = 0
        correct = 0
        with tqdm(total=len(train_loader)) as bar:
            for batch_idx, (batch_xs, target) in enumerate(train_loader):
                pred = dnn_model(batch_xs)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(pred, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                target = torch.argmax(target, -1, keepdim=False)
                batch_pred = torch.argmax(pred, -1, keepdim=False)
                correct += (target == batch_pred).sum().item()
                total += target.shape[0]

                bar.update()
                bar.set_postfix_str('loss:%.4f | avg_loss:%.4f  acc:%.4f' %
                                    (loss.item(), train_loss / (batch_idx + 1), correct / total))
        # checkpoint_path = os.path.join('./classifier', "classifier.ckpt")
        # torch.save(dnn_model, checkpoint_path)

        # Test model.
        dnn_model.eval()
        total_target = list()
        total_pred = list()

        for batch_xs, target in test_loader:
            target = torch.argmax(target, -1, keepdim=False)
            batch_out = dnn_model(batch_xs)
            batch_pred = torch.argmax(batch_out, -1, keepdim=False)

            total_target.extend(target.data.cpu().numpy())
            total_pred.extend(batch_pred.data.cpu().numpy())

        accuracy = accuracy_score(total_target, total_pred)
        precision = precision_score(total_target, total_pred)
        recall = recall_score(total_target, total_pred)
        f1 = f1_score(total_target, total_pred)

        print("Epoch %d, Test accuracy %.4f, precision %.4f, recall %.4f, f1 %.4f" %
              (epoch, accuracy, precision, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="preprocess", choices=["train", "test"])
    parser.add_argument("--epoch-idx", type=int, default=1)

    args = parser.parse_args()
    train()
