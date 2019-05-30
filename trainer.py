import os
import logging
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nlgeval import NLGEval

from dataloading import PAD_IDX
from utils import truncate

logger = logging.getLogger(__name__)


# TODO: running average
class Stats():
    def __init__(self, records):
        self.records = records
        self.reset_stats()

    def reset_stats(self):
        self.stats = {name: [] for name in self.records}

    def record_stats(self, *args):
        assert len(self.records) == len(args)
        for name, loss in zip(self.records, args):
            self.stats[name].append(loss.item())

    def report_stats(self, epoch, step='N/A'):
        to_report = {}
        for name in self.records:
            to_report[name] = np.mean(self.stats[name])
        logger.info('stats at epoch {} step {}:\n'\
                    .format(epoch, step) + str(to_report))


class EarlyStopper():
    def __init__(self, patience, metric):
        self.patience = patience
        self.metric = metric # 'Bleu_1', ..., 'METEOR', 'ROUGE_L'
        self.count = 0
        self.best_score = defaultdict(lambda: 0)
        self.is_improved = False

    def stop(self, cur_score):
        if self.best_score[self.metric] > cur_score[self.metric]:
            self.is_improved = False
            if self.count <= self.patience:
                self.count += 1
                logger.info('Counting early stop patience... {}'.format(self.count))
                return False
            else:
                logger.info('Early stopping patience exceeded. Stopping training...')
                return True # halt training
        else:
            self.is_improved = True
            self.count = 0
            self.best_score = cur_score
            return False


class Trainer():
    def __init__(self, model, data, lr,  clip, records, savedir):
        self.model = model
        self.data = data
        # TODO: implement get_optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.stats = Stats(records)
        self.savedir = savedir

    def _compute_loss(self):
        raise NotImplementedError

    def _run_epoch(self, epoch, sort_key=None, verbose=True):
        if sort_key is not None:
            self.data.train_iter.sort_key = sort_key
        for step, batch in enumerate(self.data.train_iter, 1):
            loss = self._compute_loss(batch)
            self.stats.record_stats(loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            # report train stats on a regular basis
            if verbose and (step % 100 == 0):
                self.stats.report_stats(epoch, step=step)

    def train(self):
        raise NotImplementedError

    def save_model(self, epoch):
        if not os.path.isdir(self.savdir):
            os.path.mkdir(self.savedir)
        filename = self.model.name + '_epoch{}.pt'.format(epoch)
        savedir = os.path.join(self.savedir, filename)
        torch.save(self.model.state_dict(), savedir)
        logger.info('saving model in {}'.format(savedir))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        logger.info('loading model from {}'.format(path))

    # TODO: evaluate!
    #def evaluate(self, data_type, epoch):
    #    data_iter = getattr(self.data, '{}_iter'.format(data_type))
    #    paraphrased, original, reference = [], [], []
    #    for idx, batch in enumerate(data_iter):
    #        para = self.model.inference(batch.orig)
    #        paraphrased += reverse(para, self.data.vocab)
    #        original += reverse(batch.orig[0], self.data.vocab)
    #        reference += reverse(batch.para[0], self.data.vocab)
    #    metrics_dict = self.evaluator.compute_metrics([reference], paraphrased)
    #    write_to_file(zip(original, paraphrased, reference), metrics_dict,
    #                  data_type, epoch, self.savedir)
    #    logger.info('quantitative results from {} data: '.format(data_type))
    #    print(metrics_dict)
    #    return metrics_dict


class SupervisedTrainer(Trainer):
    def __init__(self, model, data, backward=False, lr=0.001, clip=5, records=None,
                 savedir='models/'):
        super().__init__(model, data, lr, clip, records, savedir)
        self.backward = backward
        # TODO: check CE or NLL
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def _compute_loss(self, batch):
        if self.backward:
            logits = self.model(batch.resp, batch.hist2)
            target, _ = truncate(batch.hist2, 'sos')
        else:
            logits = self.model(batch.merged_hist, batch.resp)
            target, _ = truncate(batch.resp, 'sos')
        B, L, _ = logits.size()
        loss = self.criterion(logits.contiguous().view(B*L, -1),
                              target.contiguous().view(-1))
        return loss

    def train(self, num_epoch, verbose=True):
        if self.backward: # to use packed_sequence...
            sort_key = lambda ex: (len(ex.resp), len(ex.hist2))
        else:
            sort_key = None
        for epoch in range(1, num_epoch+1, 1):
            self._run_epoch(epoch, sort_key, verbose)
        self.save_model(epoch)
        return self.stats.stats


class MutulInformation():
    def __init__(self, net_forward, net_backward):
        self.net_forward = net_forward
        self.net_backward = net_backward
        self.dist = nn.CrossEntropyLoss(reduction='none') # helper

    @staticmethod
    def _calculate_prob(net, dist, x, y):
        with torch.no_grad(): # gradient no need
            logits = net(x, y)
            B, L, _ = logits.size()
            logits = logits.view(B*L, -1)
            target, lengths = y
            target = target.view(-1)
            # TODO: check dim=1
            logprob = (dist(logits, target).view(B, L).sum(dim=1) / lengths).mean()
        return logprob

    def __call__(self, batch):
        logprob_forward = self._calculate_prob(self.net_forward, self.dist,
                                               batch.merged_hist, batch.resp)
        logprob_backward = self._calculate_prob(self.net_backward, self.dist,
                                               batch.resp, batch.hist2)
        return logprob_forward + logprob_backward


# TODO: load the pretrained
class RLTrainer(Trainer):
    def __init__(self, model, data, reward, lr, to_record, clip, savedir, patience,
                 metric):
        super().__init__(model, data, lr, to_record, clip, savedir)
        self.reward = reward # callable
        self.early_stopper = EarlyStopper(patience, metric)
        self.evaluator = NLGEval(no_skipthoughts=True, no_glove=True)

    # TODO: implement reward functions
    def _compute_loss(self, batch):
        reward = self.reward(batch)
        pass

    def train(self, num_epoch):
        train_losses = []
        valid_losses, valid_metrics = [], []

        for epoch in range(1, num_epoch+1, 1):
            self._run_epoch()

            # evaluate at the end of every epoch
            with torch.no_grad():
                valid_stats = {name: [] for name in self.stats.to_record}
                for batch in self.data.valid_iter:
                    loss = self._compute_loss(batch)
                    self.stats.record_stats(loss, stat=valid_stats)
                valid_losses.append(self.stats.report_stats(epoch, stat=valid_stats))
                metrics_valid = self.evaluate('valid', epoch)
                valid_metrics.append(metrics_valid)

            if self.early_stopper is not None:
                # early stopping check
                if self.early_stopper.stop(metrics_valid):
                    self.model.load_state_dict(best_model)
                    logger.info('End of training. Best model from epoch {}'.format(best_epoch))
                    break
                elif self.early_stopper.is_improved:
                    best_model = deepcopy(self.model.state_dict())
                    best_epoch = epoch
                else: continue

        # TODO: save model to a file
        # results on test data at the end of training
        test_metrics = self.evaluate('test', epoch)

        return {'train_losses': train_losses, 'valid_losses': valid_losses,
                'valid_metrics': valid_metrics}
