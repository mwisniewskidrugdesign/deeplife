import torch
print(f'pytorch version {torch.__version__}')

num_gpu = torch.cuda.device_count()
print(f'{num_gpu} GPU available')
a=1
#  Example 1: Simple Linear Regression Task

import matplotlib.pyplot as plt
import numpy as np

Xs = np.linspace(-20,20,100)
Ys=Xs*5+3.5


a=1
import torch.nn as nn

#  we set the parameters of our simple neural network architecture
N_input = 1
N_hidden = 4
N_output = 1

#  create the model based on these parameters
model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.Linear(N_hidden,N_output)
)

#  convert our numpy arrays Xs and Ys into torch vectors we can use in training

inputs = torch.tensor(Xs.reshape((100,1)),dtype=torch.float32)
targets = torch.tensor(Ys.reshape((100,1)),dtype=torch.float32)

criterion = torch.nn.MSELoss() #  Mean Square Erroe

#  Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

N_epochs = 3000
loss_vals = []

#  Gradient Descent
for epoch in range(N_epochs):

    #  Forward pass: Compute predicted y by passing x to the model
    y_pred = model(inputs)

    # Compute and print loss
    loss = criterion(y_pred, targets)

    if epoch % 100 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())
        loss_vals.append(loss.item())

    #  Zero gradients, perform a backward pass, and update the weights.
    #  Model Parameters optimization
    optimizer.zero_grad()

    #  Perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()


a=1
test_Xs= np.linspace(-100,100,35)
test_Ys=test_Xs*5+3.5

with torch.no_grad(): # turning off the autograd of PyTorch
    test_data = torch.tensor(test_Xs.reshape((35,1)), dtype=torch.float32)
    test_output = model(test_data)

#plot the results
# plt.plot(Xs,Ys,"r.",label="training data")
# plt.plot(test_Xs,test_output,"k+",label="network testing output")
# plt.legend()
# plt.show()

