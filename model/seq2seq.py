import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH=15

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, batch_size)
        self.decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, batch_size)

    def forward(self, hist1, hist2, resp):
        encoder_output, encoder_hidden = self.encoder(hist1)
        decoder_output, decoder_hidden, decoder_attention = self.decoder()
        return decoder_output


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if embedding_weight is None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Decoder(nn.Module):

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if embedding_weight is None:
            self.embedding = nn.Embedding(out_vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(-1, 1, self.embed_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.transpose(1, 2)
        output = self.softmax(self.out(output.view(-1, self.hidden_size)))
        return output, hidden


class AttentionDecoder(nn.Module):
    """
    Apply attention based on Luong et al. (2015)
    """

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        if embedding_weight is None:
            self.embedding = nn.Embedding(out_vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs).view(self.batch_size, -1, self.embed_size)
        embedded = self.dropout(embedded)

        # step 1. GRU
        gru_out, hidden = self.gru(embedded, hidden)
        next_hidden = hidden
        # hidden.size() = (1, 40, 128)

        # print(encoder_outputs.size()) # (15, 40, 128)
        # print(hidden.size()) # (1, 40, 128)
        # hidden = hidden.transpose(0, 1)

        # step 2. socre(h_t, h_s)
        attn_prod = self.general_score(encoder_outputs, hidden)
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

        return output, next_hidden, attention_weights, out_ht

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
