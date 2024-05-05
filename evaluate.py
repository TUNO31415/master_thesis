import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

real_time_sis_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v1_score_only/"
retrospective_sis_file_path = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    # def forward(self, x, lengths):
    #     batch_size = x.size(0)
    #     # Initialize hidden state with zeros
    #     h0 = torch.zeros(batch_size, self.num_layers, self.hidden_size)
    #     c0 = torch.zeros(batch_size, self.num_layers, self.hidden_size)
    #     x = torch.unsqueeze(x,0)
    #     lengths = torch.unsqueeze(lengths, 0)
    #     # Pack padded sequence
    #     packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    #     print(f"h0_size : {h0.size()}")
    #     print(f"c0_size : {c0.size()}")
    #     # input_size = packed_x.data
    #     print(packed_x.data.shape)
    #     # Forward pass through LSTM
    #     packed_out, _ = self.lstm(packed_x, (h0, c0))
    #     # Unpack the output (padded sequence) and apply fully connected layer
    #     padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
    #     output = self.fc(padded_out[:, -1, :])  # Use the last timestep's output
    #     return output.squeeze(1)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Forward pass through LSTM
        packed_out, (hn, cn) = self.lstm(packed_x, (h0, c0))
        
        # Unpack the output (padded sequence) and apply fully connected layer
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        output = self.fc(padded_out[:, -1, :])  # Use the last timestep's output
        return output.squeeze(1)

# def run_lstm_k_fold(n_fold, X, Y): 
#     X = [torch.tensor(sublist, dtype=torch.float32) for sublist in X]
#     print(X[0])
#     Y = torch.tensor(Y, dtype=torch.float32)
#     # print(Y[:3])
#     lengths = torch.tensor([len(tensor) for tensor in X])
#     # print(lengths[0])
#     padded_X = pad_sequence(X, batch_first=True, padding_value=0.0)
#     print(padded_X[0])
#     dataset = TensorDataset(padded_X, lengths, Y)
#     kfold = KFold(n_splits=n_fold, shuffle=True)

#     input_size = 1  # Dimension of each input vector
#     hidden_size = 64  # Number of features in the hidden state of the LSTM
#     num_layers = 2  # Number of LSTM layers
#     eval_results = []

#     for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#         # Define loss function and optimizer
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_ids))
#         test_loader = torch.utils.data.DataLoader(dataset,batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_ids))

#         model = LSTMModel(input_size, hidden_size, num_layers)
#         loss_function = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#         num_epochs = 5
#         model.train()
#         for epoch in range(num_epochs):
#             running_loss = 0.0
#             for data in train_loader:
#                 optimizer.zero_grad()
#                 inputs, lengths, targets = data
#                 output = model(inputs, lengths)
#                 loss = loss_function(output, targets)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
            
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
#         model.eval()
#         test_loss = 0.0
#         with torch.no_grad():
#             for data in test_loader:
#                 inputs, lengths, targets = data
#                 outputs = model(inputs, lengths)
#                 test_loss += loss_function(outputs, targets).item()
#                 eval_results.append(evaluation_metrics(targets, outputs))

#     return eval_results

def run_lstm(X, Y): 
    X = [torch.tensor(sublist, dtype=torch.float32) for sublist in X]
    Y = torch.tensor(Y, dtype=torch.float32)
    # print(Y[:3])
    lengths = torch.tensor([len(tensor) for tensor in X])
    # print(lengths[0])
    padded_X = pad_sequence(X, batch_first=True, padding_value=0.0)
    dataset = TensorDataset(padded_X, lengths, Y)
    # kfold = KFold(n_splits=n_fold, shuffle=True)
    training_set, test_set = torch.utils.data.random_split(dataset, [0.7,0.3], generator=torch.Generator().manual_seed(42))

    print(f"x shape: {padded_X.shape}")
    print(f"lengths shape: {lengths.shape}")

    input_size = 1  # Dimension of each input vector
    hidden_size = 64  # Number of features in the hidden state of the LSTM
    num_layers = 2  # Number of LSTM layers
    eval_results = []

    model = LSTMModel(input_size, hidden_size, num_layers)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in training_set:
            optimizer.zero_grad()
            inputs, lengths, targets = data
            output = model(inputs, lengths)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_set:
            inputs, lengths, targets = data
            outputs = model(inputs, lengths)
            test_loss += loss_function(outputs, targets).item()
            eval_results.append(evaluation_metrics(targets, outputs))

    return eval_results

def peak_end_rule(X, Y):
    Y_pred = [(max(x) + x[-1])/2 for x in X]
    return evaluation_metrics(Y, Y_pred)

def peak_only(X, Y):
    Y_pred = [max(x) for x in X]
    return evaluation_metrics(Y, Y_pred)

def end_only(X, Y):
    Y_pred = [x[-1] for x in X] 
    return evaluation_metrics(Y, Y_pred)

def base_line(X, Y):
    Y_pred = [sum(x) / len(x) for x in X]
    return evaluation_metrics(Y, Y_pred)

def evaluation_metrics(Y_true, Y_pred):
    return [r2_score(Y_true, Y_pred), mean_squared_error(Y_true, Y_pred)]

def print_evaluation_result(model_name, dimension, eval_list):
    print(f"{model_name} {dimension} | R2 : {eval_list[0]} | MSE : {eval_list[1]}")

# Dimension_name = MD, CI, FI, IC, P
def data_loader(dimension_name):
    
    retro_csv_df = pd.read_csv(retrospective_sis_file_path)

    X = []
    Y = []

    for index, row in retro_csv_df.iterrows():
        batch_num = row["BatchNum"]
        selfPID = row["selfPID"]
        otherPID = row["otherPID"]

        target_file_name1 = f"score_only_rt_SIS_{selfPID}_{batch_num}_{selfPID}_{otherPID}.csv"
        target_file_name2 = f"score_only_rt_SIS_{selfPID}_{batch_num}_{otherPID}_{selfPID}.csv"
        file_name = ""
        if os.path.exists(real_time_sis_folder_path + target_file_name1):
            file_name = real_time_sis_folder_path + target_file_name1
        elif os.path.exists(real_time_sis_folder_path + target_file_name2):
            file_name = real_time_sis_folder_path + target_file_name2
        else:
            continue
        
        real_csv_df = pd.read_csv(file_name)
        try:
            x = [float(a) for a in real_csv_df[dimension_name].tolist()]
        except:
            continue
        else:
            X.append(x)
            Y.append(float(row[dimension_name]))
        
    print(f"DATASET SIZE : {len(X)}")
    return X, Y

if __name__ == "__main__":
    dimension = "MD"
    X, Y = data_loader(dimension)
    # result_pe = peak_end_rule(X, Y)
    # print_evaluation_result("PEAK END", dimension, result_pe)
    # result_p = peak_only(X, Y)
    # print_evaluation_result("PEAK ONLY", dimension, result_p)
    # result_eonly = end_only(X, Y)
    # print_evaluation_result("END ONLY", dimension, result_eonly)
    # result_base = base_line(X, Y)
    # print_evaluation_result("BASE LINE", dimension, result_base)

    res = run_lstm(X, Y)
    print(res)

    