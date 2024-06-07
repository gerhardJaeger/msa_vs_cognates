# %%
from io import open
import sys
import shutil

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import tqdm
import copy
import distance
import logging
import time
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

hidden_size = 100
epochs = 2



batch_size=100
lr = 1e-2
num_epochs = 2
log_interval = 1000


# %%

d = pd.read_csv("../data/lexibank_wordlist_pruned.csv", na_filter=False)

# %%

d["tokens"] = [" ".join(list(w)) for w in d.ASJP]

symbols = list(np.unique(np.concatenate([list(w) for w in d.ASJP])))

# %%


max_length = max(map(len, d.ASJP))+2
sos = "<sos>"
eos = "<eos>"
pad = "<pad>"
unk = "<unk>"

labels = [pad, sos, eos, unk] + symbols

vocab_size = input_size = len(labels)

idx2label = dict(enumerate(labels))
label2idx = {s:i for (i,s) in enumerate(labels)}

SOS_token = label2idx[sos]
EOS_token = label2idx[eos]

concepts = d.Concepticon_Gloss.unique()
concept2idx = {c:i for (i,c) in enumerate(concepts)}



#%%


numeric_words = [
    torch.tensor([label2idx[s] for s in [sos]+list(w)+[eos]]).view(-1, 1)    
    for w in d.ASJP
]

#%%


n_data = len(d)

train_size = int(0.7 * n_data)
val_size = int(0.1 * n_data)
test_size = n_data - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(
    numeric_words,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


#%%

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x

# %%

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size=vocab_size, hidden_size=hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
# %%

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size=hidden_size, vocab_size=vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#%%

criterion = nn.NLLLoss()
# %%

encoder = EncoderRNN().to(device)
decoder = DecoderRNN().to(device)


#%%

teacher_forcing_ratio = 0.5

#%%
def get_loss(x, encoder=encoder, decoder=decoder):
    encoder_hidden = encoder.initHidden()

    input_tensor = x.clone()
    target_tensor = x.clone()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)


    loss = 0

    for ei in range(input_length):
        _, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = StraightThroughEstimator()(encoder_hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break


    return loss.item() / target_length

#%%

def loss_estimate(val_loader):
    losses = []
    counter = 0
    for x in val_loader:
        losses.append(get_loss(x.squeeze(0).to(device)))
        counter += 1
        if counter > 100:
            break
    return np.mean(losses)


#%%


def evaluate(w, encoder=encoder, decoder=decoder, max_length=max_length):
    with torch.no_grad():
        w_tokenized = list(w)
        w_indices = [label2idx[s] for s in [sos]+w_tokenized+[eos]]
        input_tensor = torch.tensor(w_indices).view(-1, 1).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
        encoder_hidden = StraightThroughEstimator()(encoder_hidden)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(idx2label[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words

# %%

def getEmbedding(w, encoder=encoder):
    with torch.no_grad():
        w_tokenized = list(w)
        w_indices = [label2idx[s] for s in [sos]+w_tokenized+[eos]]
        input_tensor = torch.tensor(w_indices).view(-1, 1).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
        output = StraightThroughEstimator()(encoder_hidden)
        return output.view(-1).cpu().numpy()



# %%

def evWord(w):
    outputString = evaluate(w)
    wOut = "".join(outputString[1:-1])
    ds = distance.levenshtein(w, wOut)
    return ds / max(len(w), len(wOut))

#%%

def ldn_estimate(val_loader):
    output = []
    for i in np.random.choice(val_loader.dataset.indices, 100):
        output.append(evWord(d.ASJP[i]))
    return np.mean(output)


#%%

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)


    loss = 0

    for ei in range(input_length):
        _, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = StraightThroughEstimator()(encoder_hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
# %%


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#%%


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename, encoder, decoder, encoder_optimizer, decoder_optimizer):
    checkpoint = torch.load(filename)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss



#%%



def trainIters(encoder=encoder, decoder=decoder):
    tb_dir = "tensorboard_logs/"
    shutil.rmtree(tb_dir, ignore_errors=True)
    writer_training = SummaryWriter(tb_dir + "training/")
    writer_validation = SummaryWriter(tb_dir + "validation/")
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoint if available
    checkpoint_path = "checkpoint.pth"
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, encoder, decoder, encoder_optimizer, decoder_optimizer)

    for epoch in range(start_epoch, epochs):
        for iter, x in enumerate(train_loader):
            x = x.view(-1, 1).to(device)
            
            train(x, x, encoder, decoder, encoder_optimizer, decoder_optimizer)
                    
            if (iter > 0) and (iter % 1000 == 0):
                percentage = round(100 * iter / len(train_loader))
                train_loss = loss_estimate(train_loader)
                train_ldn = ldn_estimate(train_loader)
                val_loss = loss_estimate(val_loader)
                val_ldn = ldn_estimate(val_loader)
                print(f"{epoch} / {percentage}%, tls={train_loss:.4f}, vls={val_loss:.4f}, tld={train_ldn:.4f}, vld={val_ldn:.4f}")
                writer_training.add_scalar("Loss", train_loss, iter)
                writer_training.add_scalar("LDN", train_ldn, iter)
                writer_validation.add_scalar("Loss", val_loss, iter)
                writer_validation.add_scalar("LDN", val_ldn, iter)
                
                # Save checkpoint if validation loss has improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint({
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                        'loss': val_loss
                    }, filename=checkpoint_path)

# %%

trainIters()

#%%

training_results = [evWord(str(w)) for w in d.ASJP.iloc[train_dataset.indices]]
validation_results = [evWord(str(w)) for w in d.ASJP.iloc[val_dataset.indices]]

# %%

print(f"training: {round(np.mean(training_results),4)}")
print(f"validation: {round(np.mean(validation_results), 4)}")

# training: 0.0357
# validation: 0.0359

# %%

d = pd.read_csv('../data/lexibank_wordlist_pruned.csv', dtype=str).dropna(subset=["ASJP", "Cognateset_ID"])



#%%

embeddings = np.array(
    [getEmbedding(w) for w in d.ASJP],
    dtype=int)

# %%
# insert np.array(d.index) as first column into embeddings

embeddings = np.insert(embeddings, 0, np.array(d.index), axis=1)

#%%

pd.DataFrame(embeddings).to_csv("../data/embedding.csv", 
                               header=False, index=False)



# %%

