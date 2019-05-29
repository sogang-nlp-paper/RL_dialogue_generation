import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_LENGTH=22
device = torch.device('cuda')

# TODO make const file
SOS_IDX=2

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size=100, embedding_weight=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, embedding_weight)
        self.decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, embedding_weight)

    def forward(self, hist1, hist2, resp):
        resp = resp[0]
        batch_size = hist1[0].size(0)
        encoder_input = torch.cat([hist1[0], hist2[0]], dim=1)
        # encoder_outputs = (batch_size, seq_len, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input)

        # decoder
        """
        softmax_list = torch.zeros(batch_size, MAX_LENGTH).to(device)
        decoder_input = torch.tensor([batch_size * [SOS_IDX]], device=device).view(batch_size, 1)
        decoder_hidden = encoder_hidden
        print(encoder_outputs.size())
        print(resp.size())
        for i in range(MAX_LENGTH):
            softmax, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = resp[:, i]
            softmax_list[:, i] = softmax

        return softmax_list
        """


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
        embedded = self.embedding(inputs)
        output, hidden = self.lstm(embedded)
        return output, hidden


class AttentionDecoder(nn.Module):
    """ Apply attention based on Luong et al. (2015) """

    def __init__(self, out_vocab_size, embed_size, hidden_size, embedding_weight=None, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
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

    def forward(self, inputs, hidden, encoder_outputs):
        self.batch_size = inputs.size(0)
        # embedded = self.embedding(inputs).view(self.batch_size, -1, self.embed_size)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        # step 1. GRU
        lstm_out, hidden = self.lstm(embedded, hidden)
        next_hidden = hidden
        # hidden.size() = (1, 40, 128)

        # print(encoder_outputs.size()) # (15, 40, 128)
        # print(hidden.size()) # (1, 40, 128)
        # hidden = hidden.transpose(0, 1)

        # step 2. socre(h_t, h_s)
        attn_prod = self.general_score(encoder_outputs, hidden[0])
        # attn_prod = self.dot_score(encoder_outputs, hidden)

        # attention_weights = (40, 15)
        attn_prod = attn_prod.transpose(0, 1)
        attention_weights = F.softmax(attn_prod, dim=1)

        a_w = attention_weights.unsqueeze(1)

        e_o = encoder_outputs.transpose(0, 1)
        # c_t
        # context = (40, 1, 128)
        context = torch.bmm(a_w, e_o)

        # h_t = tanh(Wc[c_t;h_t])
        context = context.squeeze(1)
        hidden = hidden.squeeze(0)
        output = torch.cat((context, hidden), 1)
        output = self.attention_combine(output)
        out_ht = torch.tanh(output)  # h_tilda
        # final_hidden = output
        output = F.log_softmax(self.out(out_ht), dim=1)

        return output, next_hidden

    def general_score(self, encoder_outputs, hidden):
        """ step 2. score(h_t, h_s) general score """
        attn_prod = torch.zeros(encoder_outputs.size(0), self.batch_size, device=device)
        # print(hidden.size()) # (1, 40, 128) need transpose for bmm
        hidden = hidden.transpose(0, 1)

        # general score
        for e in range(encoder_outputs.size(0)):
            attn_prod[e] = torch.bmm(
                    hidden, self.attention(encoder_outputs[e]).unsqueeze(2)).view(self.batch_size, -1).transpose(0, 1)
        return attn_prod

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
