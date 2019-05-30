import argparse
import torch
from seq2seq import Seq2Seq
from dataloading import Data
from utils import reverse, truncate


class Simulator:

    def __init__(self, model, state_dict):
        self.model = model
        self.model.load_state_dict(torch.load(state_dict))

    def simulate(self, data):
        for batch in data.train_iter:
            softmax = self.model(batch.merged_hist, batch.resp)
            _, argmax = softmax.max(dim=2)
            print("in > " + " ".join(reverse(batch.merged_hist[0], data.vocab)))
            print("ans> " + " ".join(reverse(batch.resp[0], data.vocab)))
            print("out> " + " ".join(reverse(argmax, data.vocab)))
            print()
            print(batch.merged_hist[0].size(), batch.merged_hist[1])
            softmax = self.model.generate(batch.merged_hist)
            _, argmax = softmax.max(dim=2)
            print("in > " + " ".join(reverse(batch.merged_hist[0], data.vocab)))
            print("ans> " + " ".join(reverse(batch.resp[0], data.vocab)))
            print("out> " + " ".join(reverse(argmax, data.vocab)))
            break


if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')
    embed_size = 300

    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', default='seq2seq.bin')
    args = parser.parse_args()

    data = Data(datadir, device, batch_size=1, use_glove=False)
    vocab_size = len(data.vocab)

    model = Seq2Seq(vocab_size, embed_size, embedding_weight=data.vocab.vectors).to(device)
    simulator = Simulator(model, args.state_dict)
    simulator.simulate(data)

