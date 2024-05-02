import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, 1)  # Output is a single value
    
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

# Example usage:
# Define parameters
input_size = 1  # Dimension of each input vector
hidden_size = 64  # Number of features in the hidden state of the LSTM
num_layers = 2  # Number of LSTM layers

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data
# Assume you have a list of input sequences of varying lengths
# Each element in sequences is a tensor of shape (seq_length, input_size)
sequences = [
    torch.tensor([[1.0], [2.0], [3.0]]),
    torch.tensor([[0.5], [1.5]]),
    torch.tensor([[2.0]])
]

# Pad sequences to make them of equal length (if needed for batching)
# You can use DataLoader to handle batching and sequence padding in practice

# Training loop
model.train()
for epoch in range(num_epochs):
    for seq in sequences:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(seq.unsqueeze(0))  # Add batch dimension (1, seq_length, input_size)
        
        # Dummy target (e.g., predicting sum of the input sequence)
        target = torch.sum(seq)
        
        # Compute the loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Inference
model.eval()
test_seq = torch.tensor([[3.5], [4.5], [5.5]])
with torch.no_grad():
    predicted_value = model(test_seq.unsqueeze(0))
    print(f'Predicted value: {predicted_value.item()}')
