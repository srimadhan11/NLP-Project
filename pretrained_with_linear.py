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
from transformers import BertTokenizer, BertGenerationEncoder, BertGenerationDecoder




device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

use_pretrained_weights = False        # set this to false after first run
load_weight = False
# bert_model = 'bert-base-multilingual-cased'
bert_model = 'google/bert_uncased_L-4_H-512_A-8'

store_dir = 'model'
tokenizer_dir = os.path.join(store_dir, 'tokenizer')
encoder_dir = os.path.join(store_dir, 'encoder')
decoder_dir = os.path.join(store_dir, 'decoder')


dataset_path = './dataset2.1.csv'
trainset_path = './trainset.csv'
testset_path = './testset.csv'


def join_words(row):
    return str(row.replace("[","").replace("]","").replace("'","").replace(",",""))



# split
dataset = pd.read_csv(dataset_path, index_col=0).reset_index(drop=True)
dataset = dataset.dropna()
dataset['source'] = dataset['source'].apply(join_words)
train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)


# save
train_data.to_csv(trainset_path, index=False)
test_data.to_csv(testset_path, index=False)


# load
train_data = pd.read_csv(trainset_path, index_col=False)
test_data = pd.read_csv(testset_path, index_col=False)


if use_pretrained_weights:
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    bos_token_id, eos_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    encoder = BertGenerationEncoder.from_pretrained(bert_model, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    decoder = BertGenerationDecoder.from_pretrained(bert_model, add_cross_attention=True, is_decoder=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id)


    tokenizer.save_pretrained(tokenizer_dir)
    encoder.save_pretrained(encoder_dir)
    decoder.save_pretrained(decoder_dir)
else:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    encoder = BertGenerationEncoder.from_pretrained(encoder_dir)
    decoder = BertGenerationDecoder.from_pretrained(decoder_dir)


train_src = tokenizer.batch_encode_plus(train_data['source'], padding=True, return_tensors='pt', return_token_type_ids=False, return_attention_mask=False)['input_ids']
train_tgt = tokenizer.batch_encode_plus(train_data['target'], padding=True, return_tensors='pt', return_token_type_ids=False, return_attention_mask=False)['input_ids']


classes = list(train_data["labels"].unique())
onehot_encoder = OneHotEncoder(categories=[classes])
train_labels = onehot_encoder.fit_transform(np.array(train_data['labels']).reshape(-1, 1)).toarray()


## Hyperparameters
pad_idx = tokenizer.pad_token_id
vocab_size = len(tokenizer.vocab)
max_seq_len = tokenizer.model_max_length
n_cond_label = len(classes)
hidden_size = encoder.config.hidden_size

lr = 1e-4
batch_size = 32
epochs = 5


class Dataset(torch.utils.data.Dataset):
  def __init__(self, src, tgt, labels):
    self.src = src
    self.tgt = tgt
    self.labels = labels

  def __len__(self):
    return self.src.shape[0]

  def __getitem__(self, index):
    return self.src[index], self.tgt[index], self.labels[index]


train_dataset = Dataset(train_src, train_tgt, train_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


'''
latent: [batch, seq_len, hidden_features]
label : [batch, no_of_labels]
new_latent: [batch, seq_len, hidden_features+no_of_labels]
'''

def latent_space_concat(latent, label):
  new_label = label.unsqueeze(2).transpose(1, 2)
  new_label = new_label.expand(-1, latent.size(1), -1)
  new_latent = torch.cat((latent, new_label), dim=2)
  return new_latent


class ModifiedTransformer(nn.Module):
  def __init__(self, encoder, decoder, hidden_size, n_cond_label, tgt_vocab_size, src_pad_idx):
    super(ModifiedTransformer, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

    self.fnn1 = nn.Linear(hidden_size + n_cond_label, hidden_size*2)
    self.fnn2 = nn.Linear(hidden_size*2, hidden_size)

    self.src_pad_idx = src_pad_idx


  def forward(self, src, tgt, labels):

    attention_mask = (src == self.src_pad_idx)
    last_hidden_state = self.encoder(src, attention_mask=attention_mask).last_hidden_state

    last_hidden_state = latent_space_concat(last_hidden_state, labels)
    last_hidden_state = self.fnn1(last_hidden_state)
    last_hidden_state = F.leaky_relu(last_hidden_state)
    last_hidden_state = self.fnn2(last_hidden_state)
    last_hidden_state = F.leaky_relu(last_hidden_state)

    output = self.decoder(tgt, encoder_hidden_states=last_hidden_state, output_attentions=False, output_hidden_states=False).logits
    return output


model = ModifiedTransformer(encoder, decoder, hidden_size, n_cond_label, vocab_size, pad_idx)

if load_weight:
    model.load_state_dict(torch.load(os.path.join(store_dir, 'model.pth')))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(epochs):
    losses = np.zeros(len(train_loader), dtype=np.float32)
    model.train()
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

        del src
        del tgt
        del labels
        del predicts
        del loss

        optimizer.step()
    print(losses.tolist())
    torch.save(model.state_dict(), os.path.join(store_dir, f'model{epoch}.pth'))
    print('{}: {:.4f}'.format(epoch, losses.mean()))


torch.save(model.state_dict(), os.path.join(store_dir, 'model.pth'))

