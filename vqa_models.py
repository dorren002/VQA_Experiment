"""
Written by Kushal, modified by Robik
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from rnn import RNN


# TODO: Test with and without weight norm

class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim)
        self.dropout = nn.Dropout()
        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.use_dropout = dropout

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        if self.use_dropout:
            emb = self.dropout(emb)
        return emb


class Classifier(nn.Module):
    def __init__(self, num_input, num_hidden, num_classes, use_dropout=True):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(num_input, num_hidden)
        self.classifier = nn.Linear(num_hidden, num_classes)
        self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, feat):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(feat))
        if self.use_dropout:
            projection = self.dropout(projection)
        preds = self.classifier(projection)
        return preds


class QuestionEncoder(nn.Module):
    def __init__(self, config):
        super(QuestionEncoder, self).__init__()
        self.embedding = WordEmbedding(config.d.ntoken, config.emb_dim, config.embedding_dropout)
        self.lstm = nn.LSTM(input_size=config.emb_dim,
                            hidden_size=config.lstm_out,
                            num_layers=1, bidirectional=config.bidirectional)
        self.config = config

    #        self.lstm2 = nn.LSTM(input_size=config.emb_dim,
    #                            hidden_size=config.lstm_out,
    #                            num_layers=1, batch_first=False)

    def forward(self, q, q_len):
        q_embed = self.embedding(q)
        packed = pack_padded_sequence(q_embed, q_len, batch_first=True)
        o, (h, c) = self.lstm(packed)
        #        o1, (h1, c1) = self.lstm(q_embed)
        #        print(o1.shape)
        #        print(h1.shape)
        #        print(h.shape)
        #        o,_ = pad_packed_sequence(o)
        #        print(o.shape)
        #
        #        print("false")
        #
        #        o, (h, c) = self.lstm2(packed)
        #        o1, (h1, c1) = self.lstm2(q_embed)
        #        print(o1.shape)
        #        print(h1.shape)
        #        print(h.shape)
        #        o,_ = pad_packed_sequence(o)
        h = torch.transpose(h, 0, 1)
        return torch.flatten(h, start_dim=1)


class Attention(nn.Module):
    def __init__(self, imfeat_size, qfeat_size, use_dropout=True):
        super(Attention, self).__init__()
        self.nlin = nn.Linear(qfeat_size + imfeat_size, 1024)
        self.attnmap = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, qfeat, imfeat):
        num_objs = imfeat.size(1)
        qtile = qfeat.unsqueeze(1).repeat(1, num_objs, 1) # (1,num_obj,d_qfeat)
        #        print(qtile.shape)
        #        print(imfeat.shape)
        qi = torch.cat((imfeat, qtile), 2) # 每个个体
        qi = self.relu(self.nlin(qi))
        if self.use_dropout:
            qi = self.dropout(qi)
        attn_map = self.attnmap(qi)
        attn_map = nn.functional.softmax(attn_map, 1)
        return attn_map


class Q_only(nn.Module):
    def __init__(self, config):
        super(Q_only, self).__init__()
        assert config.use_pooled
        self.config = config
        qfeat_dim = 2048
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out

        self.classifier = Classifier(qfeat_dim,
                                     config.num_hidden,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, imfeat, ql):
        if self.config.use_lstm:
            qfeat = self.ques_encoder(q, ql)
        else:
            qfeat = q
        preds = self.classifier(qfeat)
        return preds


class UpDown(nn.Module):
    def __init__(self, config):
        super(UpDown, self).__init__() 
        assert not config.use_pooled
        self.config = config
        qfeat_dim = 2048
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out
        attention = []
        if config.attn_type == 'old':
            for i in range(config.num_attn_hops):
                attention.append(Attention(self.config.cnn_feat_size, qfeat_dim, config.attention_dropout))

        self.attention = nn.ModuleList(attention)
        self.classifier = Classifier(qfeat_dim + self.config.cnn_feat_size * config.num_attn_hops,
                                     config.num_hidden * 2,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, imfeat, ql):
        #        print(imfeat.shape)
        if self.config.use_lstm:
            qfeat = self.ques_encoder(q, ql)
        else:
            qfeat = q
        for i in range(self.config.num_attn_hops):
            attn_map = self.attention[i](qfeat, imfeat)
            scaled_imfeat = (attn_map * imfeat).sum(1)
            if i == 0:
                concat_feat = torch.cat([qfeat, scaled_imfeat], dim=1)
            else:
                concat_feat = torch.cat([concat_feat, scaled_imfeat], dim=1)
        preds = self.classifier(concat_feat)
        return preds


def main():
    pass


if __name__ == '__main___':
    main()
