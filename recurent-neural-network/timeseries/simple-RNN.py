## Simple RNN
# In this code, I'm going to train a simple RNN to do time-series prediction. Give some set
# of input data, it should be able to generate a prediction for the next time step!

## Import resources and create data

import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch import nn

plt.figure(figsize=(8, 5))

# how many time steps/data pts are in one batch of data
seq_length = 20

# generate evenly spaced dara pts
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length+1, 1))

x = data[:-1]
y = data[1:]

plt.plot(time_steps[1:], x, 'r', label='input, x')
plt.plot(time_steps[1:], y, 'b', label='target, y')

plt.legend(loc='best')
# plt.show()

## Defind the RNN

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_dim)
        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        
        output = self.fc(r_out)

        return output, hidden
    
# Check the input and output dimensions
# test that dimensions are as expected
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# generate evenly spaced test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))

test_input = torch.Tensor(data).unsqueeze(0) # give it a batch size of 1 as first dimension
print('Input size: ', test_input.size())

# test on rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())


## Training the RNN
# decide on hyperparamaters
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# Loss and Optimization

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.01)

def train(rnn, n_steps, print_every):
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        time_steps = np.linspace(step * np.pi, (step + 1)*np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1)) # input_size=1

        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        # output from the rnn
        prediction, hidden = rnn(x_tensor, hidden)


        hidden = hidden.data
        # calculate the loss
        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i%print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')
            plt.show()

    return rnn

n_steps = 75
print_every = 15

train_rnn = train(rnn, n_steps, print_every)