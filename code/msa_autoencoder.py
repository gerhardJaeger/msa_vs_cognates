#%%
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import random
import pandas as pd
import numpy as np
#%%
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#%%
db = 'dravlex'

# List the files in the directory content of msa/dravlex
alignment_files = os.listdir('msa/'+db)
alignments = [pd.read_csv('msa/' + db + '/' + fn, index_col=0) for fn in alignment_files]
#%%


# Extract all unique languages from the alignments
languages = np.unique(np.concatenate([al.index for al in alignments]))
language_to_index = {language: idx for idx, language in enumerate(languages)}
index_to_language = {idx: language for language, idx in language_to_index.items()}

#%%


# Get all symbols in the alignments
symbols = np.unique(np.concatenate([np.unique(al.values) for al in alignments]))
# Add '?' to the symbols
symbols = np.concatenate([symbols, languages, ['?']])
symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols)}
index_to_symbol = {idx: symbol for symbol, idx in symbol_to_index.items()}
#%%
sequence_length = max(al.shape[1] for al in alignments)+1

def pad_alignment(alignment, max_length):
    padded_alignment = alignment.copy()
    for i in range(max_length - alignment.shape[1]):
        padded_alignment[i + alignment.shape[1]] = '-'
    return padded_alignment

alignment_rows = []
for al in alignments:
    al_padded = pad_alignment(al, sequence_length-1)
    for i in range(al_padded.shape[0]):
        rw = np.concatenate([[al_padded.index[i]],al_padded.iloc[i].values])
        alignment_rows.append(rw)
alignment_rows = np.array(alignment_rows)
#%%
class AlignmentDataset(Dataset):
    def __init__(self, alignment_rows, symbol_to_index):
        self.alignment_rows = alignment_rows
        self.symbol_to_index = symbol_to_index

    def __len__(self):
        return len(self.alignment_rows)

    def __getitem__(self, idx):
        ar = self.alignment_rows[idx]
        
        # Encode the row using the symbol_to_index mapping
        encoded_row = [self.symbol_to_index[symbol] for symbol in ar]
        
        # Create a mask for NaNs (entries originally NaN are now '-' symbols)
        mask = np.array([symbol != '?' for symbol in ar])
        
        return torch.tensor(encoded_row, dtype=torch.int), torch.tensor(mask, dtype=torch.bool)
#%%




# Create the dataset and data loader
alignment_dataset = AlignmentDataset(alignment_rows, symbol_to_index)
train_loader = DataLoader(alignment_dataset, batch_size=4, shuffle=True)


#%%


# Define the binary autoencoder with dropout and layer normalization
class BinaryAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length, dropout_rate=0.2):
        super(BinaryAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim * sequence_length),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim * sequence_length),
            nn.Dropout(dropout_rate)
        )
        self.sequence_length = sequence_length
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.mask_token_index = vocab_size - 1  # Assuming the last index in the vocabulary is used for the mask token
        
    def forward(self, x, mask):
        # Apply dropout to the input indices to create a mask
        dropout_mask = self.dropout(torch.ones_like(x, dtype=torch.float32))
        dropout_mask = (dropout_mask > 0.5).long()
        x_masked = x * dropout_mask + (1 - dropout_mask) * self.mask_token_index

        # Use the input mask to zero out masked positions in embeddings
        embedded = self.embedding(x_masked)  # Shape: (batch_size, sequence_length, embedding_dim)
        embedded = embedded * mask.unsqueeze(-1).float()  # Zero out masked positions
        embedded = embedded.view(embedded.size(0), -1)  # Flatten the embeddings
        latent = self.encoder(embedded)  # Shape: (batch_size, latent_dim)
        binary_latent = (latent > 0).float() + latent - latent.detach()  # Differentiable binarization
        reconstruction = self.decoder(binary_latent)  # Shape: (batch_size, embedding_dim * sequence_length)
        reconstruction = reconstruction.view(reconstruction.size(0), self.sequence_length, -1)  # Reshape
        reconstruction = self.output_layer(reconstruction)  # Shape: (batch_size, sequence_length, vocab_size)
        return reconstruction, binary_latent
    
    def inference(self, x, mask):
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        embedded = embedded * mask.unsqueeze(-1).float()  # Zero out masked positions
        embedded = embedded.view(embedded.size(0), -1)  # Flatten the embeddings
        latent = self.encoder(embedded)  # Shape: (batch_size, latent_dim)
        binary_latent = (latent > 0).float()  
        return binary_latent
    
#%%

# Hyperparameters
vocab_size = len(symbol_to_index)
embedding_dim = 64
hidden_dim = 512
latent_dim = 256
num_epochs = 100
learning_rate = 0.0001
dropout_rate = 0.1
#%%


# Initialize the model, loss function, and optimizer
model = BinaryAutoencoder(vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#%%

# Define accuracy function
def calculate_accuracy(reconstruction, target, mask):
    _, predicted = torch.max(reconstruction, 2)
    predicted_nonmask = predicted[mask].to('cpu').numpy()
    target_nonmask = target[mask].to('cpu').numpy()
    accuracy = (predicted_nonmask[predicted_nonmask != '-'] == target_nonmask[predicted_nonmask != '-']).mean()
    return accuracy
#%%
for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    for encoded_row, mask in train_loader:
        # Move data to GPU
        encoded_row = encoded_row.to(device)
        mask = mask.to(device)
        
        # Forward pass
        reconstruction, binary_latent = model(encoded_row, mask)
        reconstruction = reconstruction.view(-1, vocab_size)  # Reshape for loss calculation
        encoded_row = encoded_row.view(-1).long()  # Flatten target and convert to Long type
        
        # Apply mask to ignore invalid positions in loss calculation
        mask = mask.view(-1)
        loss = criterion(reconstruction[mask], encoded_row[mask])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        reconstruction = reconstruction.view(-1, sequence_length, vocab_size)
        encoded_row = encoded_row.view(-1, sequence_length)
        accuracy = calculate_accuracy(reconstruction, encoded_row, mask.view(-1, sequence_length))
        total_accuracy += accuracy
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')
    
#%%

inference_loader = DataLoader(alignment_dataset, batch_size=4, shuffle=False)

embeddings_ = []
for encoded_row, mask in inference_loader:
    encoded_row = encoded_row.to(device)
    mask = mask.to(device)
    binary_latent = model.inference(encoded_row, mask)
    embeddings_.append(binary_latent.detach().cpu().numpy())

# %%
embeddings = np.concatenate(embeddings_).astype(int)
# %%
