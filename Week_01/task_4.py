import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#  Overfitting by undersampling

cos_Xs=np.linspace(0,10,100)
cos_Ys=np.cos(cos_Xs)
#  plt.plot(cos_Xs,cos_Ys)

#  Train Set
train_cos_Xs = np.random.rand(200)*6
train_cos_Ys = np.cos(train_cos_Xs)
#  Validation Set
val_cos_Xs=np.random.rand(100)*6
val_cos_Ys = np.cos(val_cos_Xs)

#  we set the parameters of our simple network architecture
N_input = 1
N_hidden = 25
N_output = 1

#  create the model based on these parameters
model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_output),
)
#  convert our numpy arrays Xs and Ys into torch vectors we can use in training
inputs = torch.tensor(train_cos_Xs.reshape((200,1)), dtype=torch.float32)
targets = torch.tensor(train_cos_Ys.reshape((200,1)), dtype=torch.float32)

#  validation inputs and targets
val_inputs = torch.tensor(val_cos_Xs.reshape((100,1)), dtype = torch.float32)
val_targets = torch.tensor(val_cos_Ys.reshape((100,1)), dtype=torch.float32)

criterion = torch.nn.MSELoss()  # Mean Square Error

#  construct the optimizer (Now we use Adam, a bit smarter optimizer)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epochs = 5000
loss_vals = []
valid_vals = []

#  Gradient Descent
for epoch in range(N_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(inputs)

    #  Compute and print loss
    loss = criterion(y_pred,targets)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    # perform a backward pass (backpropagation)
    loss.backward()
    # Update the parameters
    optimizer.step()

    if epoch % 500 == 0:
        loss_vals.append(loss.item())
        with torch.no_grad():
            val_output = model(val_inputs)
            val_loss = criterion(val_output, val_targets)
            valid_vals.append(val_loss.item())
            print('epoch: ', epoch, ' loss: ', loss.item(), "validation loss: ", val_loss.item())

#let us see the behavior of both training and validation loss
plt.plot(loss_vals,label="training loss")
plt.plot(valid_vals,label="validation loss")
plt.legend()
plt.show()