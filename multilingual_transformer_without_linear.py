import os
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer



nlp_path     = '/data/madhan/files/nlp'
bert_model   = 'bert-base-multilingual-cased'
default_cuda = 2
use_saved_weights   = False
use_saved_dataset   = True
use_saved_tokenizer = True
classes = ['FRENCH', 'SPANISH', 'NON-HUMOROUS', 'HUMOROUS']


project_path  = os.path.join(nlp_path, 'project/p1')
model_dir     = os.path.join(project_path, 'model')
tokenizer_dir = os.path.join(project_path, 'tokenizer')
dataset_path  = os.path.join(project_path, 'dataset1.csv')
trainset_path = os.path.join(project_path, 'trainset.csv')
testset_path  = os.path.join(project_path, 'testset.csv')


torch.cuda.set_device(default_cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def tokenize(tokenizer, sequence):
    ret = tokenizer.batch_encode_plus(sequence, padding=True, truncation=True, return_tensors='pt',
                                      return_token_type_ids=False, return_attention_mask=False)
    return ret['input_ids']



class Dataset(torch.utils.data.Dataset):
    def __init__(self, src, tgt, labels):
        self.src = src
        self.tgt = tgt
        self.labels = labels

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], self.labels[index]



class ModifiedTransformer(nn.Module):
    def __init__(self, hidden_size = 512, nhead = 8, n_cond_label = 1,
                         num_encoder_layers = 6, num_decoder_layers = 6,
                         dim_feedforward = 2048, dropout = 0.1,
                         src_pad_idx = 0, src_vocab_size = 30000, tgt_vocab_size=30000,
                         max_src_len = 100, max_tgt_len = 100):
        super(ModifiedTransformer, self).__init__()

        self.hidden_size  = hidden_size
        self.n_cond_label = n_cond_label
        self.src_pad_idx  = src_pad_idx
        self.max_src_len  = max_src_len
        self.max_tgt_len  = max_tgt_len

        self.src_pe        = ModifiedTransformer.get_pe(max_src_len, hidden_size)
        self.tgt_pe        = ModifiedTransformer.get_pe(max_src_len, hidden_size+n_cond_label)
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_size+n_cond_label)

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout, 'relu')
        encoder_norm  = nn.LayerNorm(hidden_size)
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(hidden_size+n_cond_label, nhead, dim_feedforward, dropout, 'relu')
        decoder_norm  = nn.LayerNorm(hidden_size+n_cond_label)
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.fnn_out = nn.Linear(hidden_size+n_cond_label, tgt_vocab_size)
        self._reset_parameters()


    def forward(self, src, tgt, labels):
        src_N, src_seq_len           = src.shape
        tgt_N, tgt_seq_len           = tgt.shape
        labels_N, labels_feature_len = labels.shape

        test = "number of batches in src, tgt and labels must be equal"
        assert src_N == tgt_N == labels_N, test

        test = "number of label features is not same as n_cond_label"
        assert labels_feature_len == self.n_cond_label, test


        emb_src = self.src_embedding(src) + self.src_pe[:src_seq_len, :].to(src.device)
        emb_tgt = self.tgt_embedding(tgt) + self.tgt_pe[:tgt_seq_len, :].to(src.device)

        src_key_padding_mask = (src == self.src_pad_idx)
        tgt_mask = ModifiedTransformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        memory = self.encoder(emb_src.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        new_memory = ModifiedTransformer.latent_space_concat(memory.transpose(0, 1), labels).transpose(0, 1)
        output = self.decoder(emb_tgt.transpose(0, 1), new_memory, tgt_mask=tgt_mask)
        output = self.fnn_out(output.transpose(0, 1))
        return output


    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    @staticmethod
    def latent_space_concat(latent, label):
        '''
        latent: [batch, seq_len, hidden_features]
        label : [batch, no_of_labels]
        new_latent: [batch, seq_len, hidden_features+no_of_labels]
        '''
        new_label = label.unsqueeze(2).transpose(1, 2)
        new_label = new_label.expand(-1, latent.size(1), -1)
        new_latent = torch.cat((latent, new_label), dim=2)
        return new_latent


    @staticmethod
    def get_pe(max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe          = torch.empty(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)


def main():
    if use_saved_dataset:
        # load previously saved
        train_data = pd.read_csv(trainset_path, index_col=False)
        test_data = pd.read_csv(testset_path, index_col=False)
    else:
        # load entire dataset
        dataset = pd.read_csv(dataset_path, index_col=0)
        dataset = dataset.dropna()

        # split
        train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)

        # save
        train_data.to_csv(trainset_path, index=False)
        test_data.to_csv(testset_path, index=False)


    if use_saved_tokenizer:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        tokenizer.save_pretrained(tokenizer_dir)


    train_src = tokenize(tokenizer, train_data['source'])
    train_tgt = tokenize(tokenizer, train_data['target'])
    onehot_encoder = OneHotEncoder(categories=[classes])
    train_labels = onehot_encoder.fit_transform(train_data['labels'].to_numpy().reshape(-1, 1)).toarray()


    # Hyperparameters
    lr         = 1e-4
    batch_size = 2
    epochs     = 15

    vocab_size     = len(tokenizer.vocab)
    max_seq_len    = tokenizer.model_max_length
    pad_idx        = tokenizer.pad_token_id
    src_vocab_size = tgt_vocab_size = vocab_size
    max_src_len    = max_tgt_len    = max_seq_len

    hidden_size       = 512
    nhead             = 1
    n_cond_label      = len(classes)
    num_encoder_layer = 6
    num_decoder_layer = 6
    dim_feedforward   = 2048
    dropout           = 0.1


    # dataloader
    train_dataset = Dataset(train_src, train_tgt, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # prepare model
    model = ModifiedTransformer(hidden_size, nhead, n_cond_label, num_encoder_layer, num_decoder_layer,
                                dim_feedforward, dropout, pad_idx, src_vocab_size, tgt_vocab_size,
                                max_src_len, max_tgt_len)
    if use_saved_weights:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    model = model.to(device)

    # optimizer and objective function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # train
    for epoch in range(epochs):
        model.train()
        losses = np.zeros(len(train_loader), dtype=np.float32)
        for idx,(src,tgt,labels) in enumerate(train_loader):
            optimizer.zero_grad()

            src = torch.as_tensor(src, dtype=torch.int64).to(device)
            tgt = torch.as_tensor(tgt, dtype=torch.int64).to(device)
            labels = torch.as_tensor(labels, dtype=torch.float32).to(device)

            predicts = model(src, tgt[:, :-1], labels)
            predicts = F.log_softmax(predicts, dim=2)
            predicts = predicts.reshape(-1, predicts.shape[2])
            loss = criterion(predicts, tgt[:, 1:].reshape(-1))
            loss.backward()
            losses[idx] = loss.item()

            optimizer.step()
        torch.save(model.state_dict(), os.path.join(model_dir, f'model{epoch}.pth'))
        print('{}: {:.4f}'.format(epoch, losses.mean()))
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

if __name__ == '__main__':
    main()
