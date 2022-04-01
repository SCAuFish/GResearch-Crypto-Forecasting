import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import List
from data_loader import AssetSerialData


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, fc_dim: List,
                 dropout_prob, bidirectional=True):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob,
            bidirectional=bidirectional
        )

        # Fully connected layer
        self.fcs = nn.ModuleList()
        fc_dim.insert(0, hidden_dim * (2 if bidirectional else 1))
        fc_dim.append(output_dim)
        for i in range(len(fc_dim) - 1):
            input_dim = fc_dim[i]
            output_dim = fc_dim[i+1]
            self.fcs.append(nn.BatchNorm1d(input_dim))
            self.fcs.append(nn.Linear(input_dim, output_dim))
            self.fcs.append(nn.Tanh())

        self.init_hidden()

    def init_hidden(self):
        dimension = self.layer_dim * (2 if self.bidirectional else 1)
        self.hidden = nn.Parameter(torch.zeros(dimension, 1, self.hidden_dim))
        self.cell_state = nn.Parameter(torch.zeros(dimension, 1, self.hidden_dim))

    def forward(self, x: torch.tensor, logging=False):
        x = x.float()
        h0 = self.hidden
        c0 = self.cell_state
        batch_size = x.shape[0]
        h0 = torch.cat([h0 for i in range(batch_size)], dim=1)
        c0 = torch.cat([c0 for i in range(batch_size)], dim=1)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        # if logging:
        #     print(out)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        for layer in self.fcs:
            out = layer(out)
            if logging:
                print(layer)
                if type(layer) == nn.Linear:
                    print(f"linear layer weight average: {torch.mean(torch.abs(layer.weight))}")
                    print(layer.weight)
                    print(layer.bias)
                print(out)

        return out


def train(data_loader, model, optim, loss_fn, device):
    total_loss = 0
    for batch in tqdm(data_loader):
        batch = batch.to(device).float()
        optim.zero_grad()
        x, t = batch[:, :, 1:], batch[:, -1, 0]
        y = model(x, logging=False)
        # print(f"input: {x}")
        # print(f"prediction: {y}")
        # print(f'target: {t}')

        loss = loss_fn(y, t.reshape((-1, 1)))
        loss.backward()
        optim.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model, data_loader, device):
    with torch.no_grad():
        total_loss = 0
        prediction = []
        target = []
        for batch in data_loader:
            batch = batch.to(device)
            x, t = batch[:, :, 1:], batch[:, -1, 0]
            pred = model(x)

            loss = loss_fn(pred, t)
            total_loss += loss

            prediction.append(pred)
            target.append(t)

    all_prediction = torch.cat(prediction, dim=0)
    all_target = torch.cat(target, dim=0)
    combined = torch.cat([all_prediction, all_target.view(-1, 1)], dim=1).transpose(0, 1)
    corr = torch.corrcoef(combined)
    return corr


def is_better_model(loss, corr, min_loss, max_corr):
    if min_loss > loss or max_corr < corr:
        return True
    else:
        return False


def save_checkpoint(model, epoch, name):
    torch.save(model.state_dict(), f"../model/{name}-{epoch}.bin")


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# torch.distributed.init_process_group()

# data_filename = "../data/combined-processed-time-series-train-length-5.pkl"
# dataset = AssetSerialData(data_filename, 5)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
epoch = 500

test_data_filename = "../data/combined-processed-time-series-test-length-5.pkl"
test_dataset = AssetSerialData(test_data_filename, 5)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LSTMModel(
    input_dim=300, layer_dim=2, hidden_dim=128, fc_dim=[128, 128, 128, 64], output_dim=1, dropout_prob=0.1,
    bidirectional=False).to(device)
# print(model)
optim = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

min_loss = 1000
max_corr = -1
for e in range(epoch):
    print(f"Training {e}th epoch")
    # loss = train(data_loader, model, optim, loss_fn, device=device)
    loss = train(test_data_loader, model, optim, loss_fn, device=device)
    print(f"loss: {loss}")

    corr = evaluate(model, test_data_loader, device=device)
    corr = corr[0][1]
    print(f"corr coef: {corr}")
    #
    # if is_better_model(loss, corr, min_loss, max_corr):
    #     print(f"saving model")
    #     save_checkpoint(model, e, "lstm-baseline")
    #     min_loss = loss
    #     max_corr = corr
