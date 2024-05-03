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
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, 1)  # Output is a single value
    
    def forward(self, x, lengths):
        # Sort input sequences by length (required for pack_padded_sequence)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sorted_idx]
        
        # Pack padded sequences
        packed_x = pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True, enforce_sorted=True)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        packed_out, _ = self.lstm(packed_x, (h0, c0))
        
        # Unpack and pad packed sequence
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Index into the padded sequence to get the last hidden state of each sequence
        # Note: Use original indices to recover the original order of sequences
        idx_unsorted = torch.argsort(sorted_idx)
        last_hidden_states = padded_out[idx_unsorted, sorted_lengths - 1, :]
        
        # Apply fully connected layer
        out = self.fc(last_hidden_states)
        
        return out.squeeze(1)

def run_lstm(n_fold, X, Y): 
    X = [torch.tensor([[a] for a in x]) for x in X]
    Y = [torch.tensor([[y]]) for y in Y]
    # print(Y[:3])
    lengths = [torch.tensor([seq.size(0) for seq in X])]
    print(lengths[:3])
    padded_X = pad_sequence(X, batch_first=True, padding_value=0.0)
    dataset = TensorDataset(padded_X, lengths, Y)
    kfold = KFold(n_splits=n_fold, shuffle=True)

    input_size = 1  # Dimension of each input vector
    hidden_size = 64  # Number of features in the hidden state of the LSTM
    num_layers = 2  # Number of LSTM layers
    eval_results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Define loss function and optimizer
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset,batch_size=10, sampler=test_subsampler)

        model = LSTMModel(input_size, hidden_size, num_layers)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5
        model.train()
        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                inputs, lengths, targets = data
                # Forward pass
                output = model(inputs, lengths)  # Add batch dimension (1, seq_length, input_size)
                loss = loss_function(output, targets)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
        model.train(False)

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, targets = data
                outputs = model(inputs)
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

    res = run_lstm(3, X, Y)
    print(res)

    