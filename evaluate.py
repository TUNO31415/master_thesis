import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

real_time_sis_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v1_score_only/"
retrospective_sis_file_path = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, 1)
        
        return out.squeeze(1)  # Squeeze to make output shape (batch_size,)

def run_lstm(dimension_name, n_fold): 
    X, Y = data_loader(dimension_name)
    X = [torch.tensor(x) for x in X]
    Y = [torch.tensor(y) for y in Y]

    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True)

    input_size = 1  # Dimension of each input vector
    hidden_size = 64  # Number of features in the hidden state of the LSTM
    num_layers = 2  # Number of LSTM layers
    model = LSTMModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    eval_results = []

    for train, test in kfold.split(X, Y):
        # Define loss function and optimizer
        num_epochs = 5
        model.train()
        for epoch in range(num_epochs):
            for x in X[train]:
                optimizer.zero_grad()
                # Forward pass
                output = model(x.unsqueeze(0))  # Add batch dimension (1, seq_length, input_size)    
                loss = criterion(output, Y[train])
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
        model.train(False)
        # Y_pred = []
        # with torch.no_grad():
        #     predicted_value = model(test_seq.unsqueeze(0))
        #     print(f'Predicted value: {predicted_value.item()}')
        Y_pred = model(X[test])
        eval_results.append(evaluation_metrics(Y[test], Y_pred))

    return eval_results()

def peak_end_rule(dimension_name):
    X, Y_true = data_loader(dimension_name)
    Y_pred = [(max(x) + x[-1])/2 for x in X]
    return evaluation_metrics(Y_true, Y_pred)

def peak_only(dimension_name):
    X, Y_true = data_loader(dimension_name)
    Y_pred = [max(x) for x in X]
    return evaluation_metrics(Y_true, Y_pred)

def end_only(dimension_name):
    X, Y_true = data_loader(dimension_name)
    Y_pred = [x[-1] for x in X] 
    return evaluation_metrics(Y_true, Y_pred)

def base_line(dimension_name):
    X, Y_true = data_loader(dimension_name)
    Y_pred = [sum(x) / len(x) for x in X]
    return evaluation_metrics(Y_true, Y_pred)

def evaluation_metrics(Y_true, Y_pred):
    return r2_score(Y_true, Y_pred), mean_squared_error(Y_true, Y_pred)

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

        file_name = target_file_name1 if os.path.exists(real_time_sis_folder_path + target_file_name1) else target_file_name2

        real_csv_df = pd.read_csv(file_name)
        X.append(real_csv_df[dimension_name].tolist())
        Y.append(row[dimension_name])
    
    return X, Y

if __name__ == "__main__":
    dimension = "MD"
    result_pe_rule = peak_end_rule(dimension)
    print(result_pe_rule)