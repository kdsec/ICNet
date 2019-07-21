import torch

from model import QuickThoughtsModel

WORD_EMBED = 128
THOUGHT_SIZE = 256


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(1234)
    vocab_size = 5116

    model_path = 'quick_thought_main_all.bin'
    print('    load model', model_path)
    ckpt = torch.load(model_path)

    model = QuickThoughtsModel(vocab_size, WORD_EMBED, THOUGHT_SIZE,
                               1, dropout=0,
                               bidirectional=False,
                               pretrained_weight=None,
                               device=device)
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    main()
