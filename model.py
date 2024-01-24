from torch import nn
import torch
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, layers_size, dropout_prob) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers_size = layers_size
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout_prob)
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, layers_size, bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size
        
        x = self.embedding(x)
        x = self.dropout(x)
        # x shape: (seq_length, N, embedding_size)
        
        encoder_states, (hidden, cell) = self.lstm(x)
        # x shape: (seq_length, N, hidden_size)
        
        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        
        return encoder_states, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, layers_size, dropout_prob) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers_size = layers_size
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout_prob)
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(hidden_size * 2 + embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        
    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        lstm_input = torch.cat((context_vector, embedding), dim=2)
        # lstm_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, len_english_vocab, teacher_forcing_ratio=0.7):
        self.batch_size = source.shape[1]
        self.target_len = target.shape[0]
        self.target_vocab_size =  len_english_vocab
        outputs = torch.zeros(self.target_len, self.batch_size, self.target_vocab_size)
        
        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]
        
        for i in range(1, self.target_len):
            perdiction, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[i] = perdiction
            best_guess = perdiction.argmax(1)
            x = target[i] if random.random() < teacher_forcing_ratio else best_guess
        
        return outputs
    
    def predict(self, x):
        x = self.encoder.embedding(x)
        x = self.encoder.lstm(x)
        x = self.decoder.lstm(x)
        return x
    
    def save_checkpoint(self, state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])