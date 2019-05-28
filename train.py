import torch
import torch.nn as nn
import torch.optim as optim
from model.seq2seq import Seq2Seq
from dataloading import Data


class Trainer():

    def __init__(self, model, lr=0.001):
        self.lr = lr
        self.model = model
        self.loss_function = nn.NLLLoss()
        self.optim = optim.Adam(self.model.parameters(), lr)

    def train(self, data, epoch=10):
        for ep in range(epoch):
            for batch in data.train_iter:
                print(batch)
                # softmax = self.model(batch)
                # print(softmax)

    def save_model(self, state_dict_name='model.bin'):
        torch.save(self.model.state_dict(), state_dict_name)


if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')
    embed_size=300

    data = Data(datadir, device, batch_size=2, use_glove=True)
    print(data.vocab.freqs)
    print(data.vocab.stoi)
    print(data.vocab.itos)
    vocab_size = len(data.vocab)
    print(len(data.vocab.itos), len(data.vocab))

    trainer = Trainer(Seq2Seq(vocab_size, embed_size, hidden_size=100, batch_size=2))
    trainer.train(data, epoch=10)
    # trainer.save_model('seq2seq.bin')

