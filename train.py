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
                softmax = self.model(batch.merged_hist, batch.resp)
                # print(softmax)

    def save_model(self, state_dict_name='model.bin'):
        torch.save(self.model.state_dict(), state_dict_name)


if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')
    embed_size=300

    data = Data(datadir, device, batch_size=2, use_glove=False)
    vocab_size = len(data.vocab)

    model = Seq2Seq(vocab_size, embed_size, embedding_weight=data.vocab.vectors).to(device)
    trainer = Trainer(model)
    trainer.train(data, epoch=10)
    # trainer.save_model('seq2seq.bin')

