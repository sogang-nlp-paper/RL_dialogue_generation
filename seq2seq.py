import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from utils import truncate
from dataloading import PAD_IDX, MAXLEN

device = torch.device('cuda')


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size=100, embedding_weight=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, embedding_weight)
        self.decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, embedding_weight)

    def forward(self, merged_hist, resp):
        encoder_inputs = truncate(merged_hist, 'sos')
        encoder_outputs, encoder_hidden = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(resp[0], encoder_hidden, encoder_inputs[0], encoder_outputs)
        return decoder_outputs


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, embedding_weight=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        if embedding_weight is None:
            self.embedding = nn.Embedding(vocab_size, self.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        sentence_tensor, length = inputs
        length.cpu()
        embedded = self.embedding(sentence_tensor)
        packed = pack_padded_sequence(embedded, length.cpu(), batch_first=True)
        output, hidden = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden


class AttentionDecoder(nn.Module):
    """ Apply attention based on Luong et al. (2015) """

    def __init__(self, out_vocab_size, embed_size, hidden_size, embedding_weight=None, dropout_p=0.1, max_length=MAXLEN):
        super(AttentionDecoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = 1

        if embedding_weight is None:
            self.embedding = nn.Embedding(out_vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, decoder_inputs, hidden, encoder_inputs, encoder_outputs):
        batch_size, seq_len = decoder_inputs.size(0), decoder_inputs.size(1)
        softmax_matrix = torch.zeros(seq_len, batch_size, self.out_vocab_size).to(device)

        for i in range(seq_len):
            embedded = self.embedding(decoder_inputs[:, i]).unsqueeze(1)

            # step 1. lstm
            lstm_out, hidden = self.lstm(embedded, hidden)
            h_t = hidden[0].squeeze(0)  # hidden = (batch, hidden)

            # step 2. attention socre(h_t, h_s)
            attn_weight = self.general_score(encoder_inputs, encoder_outputs, h_t)
            # attn_weight = self.dot_score(encoder_outputs, hidden)

            # c_t
            # context = (batch_size, 1, hidden_size)
            context = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)

            # h_t = tanh(Wc[c_t;h_t])
            context = context.squeeze(1)
            output = torch.cat((context, h_t), dim=1)
            out_ht = torch.tanh(self.attention_combine(output))  # h_tilda

            softmax_output = F.softmax(self.out(out_ht), dim=1)  # (batch_size, vocab_size)
            softmax_matrix[i] = softmax_output

        return softmax_matrix.transpose(0, 1)

    def general_score(self, encoder_inputs, encoder_outputs, ht):
        """ step 2. score(h_t, h_s) general score """
        w_hs = self.attention(encoder_outputs)
        ht = ht.unsqueeze(2)
        attn_prod = torch.bmm(w_hs, ht).squeeze(2)
        attn_prod.masked_fill(encoder_inputs == PAD_IDX, 0)
        attn_weight = F.softmax(attn_prod, dim=1)
        return attn_weight  # (batch_size, seq_len)

    def dot_score(self, encoder_outputs, hidden):
        """ step 2. score(h_t, h_s) dot score """
        attn_prod = torch.zeros(encoder_outputs.size(0), self.batch_size, device=device)
        # print(hidden.size()) # (1, 40, 128) need transpose for bmm
        hidden = hidden.transpose(0, 1)

        # dot score
        for e in range(encoder_outputs.size(0)):
            attn_prod[e] = torch.bmm(
                    hidden, encoder_outputs[e].unsqueeze(2)).view(self.batch_size, -1).transpose(0, 1)
        return attn_prod
