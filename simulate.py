import argparse
import torch
from seq2seq import Seq2Seq
from dataloading import Data
from utils import reverse, truncate, concat
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda')


class Simulator:

    def __init__(self, model, state_dict=None):
        self.agentA = model
        self.agentB = model
        if state_dict is not None:
            self.agentA.load_state_dict(torch.load(state_dict))
            self.agentB.load_state_dict(torch.load(state_dict))
        self.agent = [self.agentA, self.agentB]

    def simulate(self, data, turn=3):
        for batch in data.train_iter:
            input_message = batch.hist1
            history1 = input_message
            for t in range(turn):
                agent = self.agent[(t % 2)]
                decoder_out = agent.generate(input_message)  # type of decoder out = [data, lenght]

                history2 = decoder_out
                input_message = concat(history1, history2)
                history1 = history2

    def debug(self, data, turn=3, sample_num=10):
        logger.info("Debugging ...")
        for i, batch in enumerate(data.train_iter):
            assert len(batch) == 1, 'batch size must be 1 in debugging'
            input_message = batch.hist1
            history1 = input_message[0]
            logger.info(" sample %d ", (i+1))
            for t in range(turn):
                agent = self.agent[(t % 2)]
                logits_matrix = agent.generate(input_message)
                _, decoder_out = logits_matrix.max(dim=2)
                logger.info("[turn %d] IN : " + " ".join(reverse(input_message[0], data.vocab)), (t+1))
                logger.info("[turn %d] OUT: " + " ".join(reverse(decoder_out, data.vocab)), (t+1))

                history2 = decoder_out
                input_message = torch.cat([history1, history2], dim=1)  # TODO concat without padding
                input_message = [input_message, torch.LongTensor([input_message.size(1)])]  # TODO need to pack with batch
                history1 = history2
            if (i+1) == sample_num:
                break


if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')
    embed_size = 300

    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', default='forward_epoch1.pt')
    args = parser.parse_args()

    data = Data(datadir, device, batch_size=1, use_glove=False)
    vocab_size = len(data.vocab)

    model = Seq2Seq(vocab_size, embed_size, embedding_weight=data.vocab.vectors).to(device)
    simulator = Simulator(model, args.state_dict)
    # simulator.debug(data)
    simulator.simulate(data)

