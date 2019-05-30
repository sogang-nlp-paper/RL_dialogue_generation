import logging
from setproctitle import setproctitle

import torch

from dataloading import Data
from seq2seq import Seq2Seq
from trainer import SupervisedTrainer, RLTrainer
#from utils import kl_coef, plot_kl_loss, plot_learning_curve, plot_metrics

setproctitle("(hwijeen) RL dialogue")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":
    DATA_DIR = 'data/'
    DEVICE = torch.device('cuda:0')
    EPOCH = 1

    data = Data(DATA_DIR, DEVICE, batch_size=64, use_glove=False)
    seq2seq = Seq2Seq(len(data.vocab), 300, 500, name='forward').to(DEVICE)
    seq2seq_back = Seq2Seq(len(data.vocab), 300, 500, name='backward').to(DEVICE)
    trainer_seq = SupervisedTrainer(seq2seq, data, lr=0.001, records=['NLLLoss'])
    trainer_back = SupervisedTrainer(seq2seq_back, data, lr=0.001, records=['NLLLoss'],
                                     backward=True)
    results_seq = trainer_seq.train(num_epoch=EPOCH, verbose=True)
    results_back = trainer_back.train(num_epoch=EPOCH, verbose=True)

    #plot_learning_curve(results['train_losses'], results['valid_losses'])
    #plot_metrics(results['train_metrics'], results['valid_metrics'])
    #plot_kl_loss(trainer.stats.stats['kl_loss'])

