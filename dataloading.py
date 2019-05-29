import os
import logging

import torch
from torchtext.data import Field, TabularDataset, BucketIterator


MAXLEN = 22
logger = logging.getLogger(__name__)

# TODO: check if correct
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
# TODO: <SEP> needed?


class Data(object):
    def __init__(self, data_dir, device, batch_size, use_glove=False):
        self.device = device
        self.train_path = os.path.join(data_dir, 'train.txt')
        self.test_path = os.path.join(data_dir, 'test.txt')
        self.build(use_glove, batch_size)
        self.report_stats()

    def report_stats(self):
        logger.info('data size... {} / {} / {}'.\
                    format(len(self.train), len(self.val), len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build(self, use_glove, batch_size):
        logger.info('building field, dataset, vocab, dataiter...')
        HIST1, HIST2, RESP = self.build_field(maxlen=MAXLEN)
        self.train, self.val, self.test = self.build_dataset(HIST1, HIST2, RESP)
        sources = [self.train.hist1, self.train.hist2, self.train.resp,
                   self.val.hist1, self.val.hist2, self.val.resp]
        self.vocab = self.build_vocab(HIST1, HIST2, RESP, sources, use_glove)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test, batch_size)

    def build_field(self, maxlen=None):
        HIST1 = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        init_token='<sos>', eos_token='<eos>', tokenize='toktok')
        HIST2 = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        init_token='<sos>', eos_token='<eos>', tokenize='toktok')
        RESP = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        init_token='<sos>', eos_token='<eos>', tokenize='toktok')
        return HIST1, HIST2, RESP

    def build_dataset(self, HIST1, HIST2, RESP):
        train_val = TabularDataset(path=self.train_path, format='tsv',
                                fields=[('hist1', HIST1), ('hist2', HIST2),
                                        ('resp', RESP)])
        train, val = train_val.split(split_ratio=0.8)
        test = TabularDataset(path=self.test_path, format='tsv',
                                fields=[('hist1', HIST1), ('hist2', HIST2),
                                        ('resp', RESP)])
        for data in [train, val, test]: # merging hist1 and hist2
            data.fields['merged_hist'] = data.fields['hist1']
            for ex in data.examples:
                setattr(ex, 'merged_hist', ex.hist1 + ex.hist2)
        return train, val, test

    def build_vocab(self, HIST1, HIST2, RESP, sources, use_glove=False):
        v = 'glove.6B.300d' if use_glove else None
        HIST1.build_vocab(sources, max_size=30000, vectors=v)
        HIST2.vocab = RESP.vocab = HIST1.vocab
        return HIST1.vocab

    def build_iterator(self, train, val, test, batch_size=32):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=batch_size,
                              sort_key=lambda ex: (len(ex.meged_hist),
                                                   len(ex.hist1), len(ex.hist2),
                                                   len(ex.resp)),
                              sort_within_batch=True, repeat=False,
                              device=self.device)
        return train_iter, valid_iter, test_iter

if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')

    data = Data(datadir, device, batch_size=2, use_glove=False)

    for batch in data.train_iter:
        print(batch)
    print(data.vocab.itos)


