import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


#  Task 2 - we have 2 arguments

Xs1 = np.linspace(-10,10,50)
Xs2 = np.linspace(-10,10,50)

Xs1,Xs2 = np.meshgrid(Xs1,Xs2) # generate all possible pairs of x1 and x2
Ys = (Xs2**2*5)-(Xs1*3) + 15
a=1

#preparing torch tensors for learning
inputs = torch.tensor(np.concatenate((Xs1.reshape((2500,1)),Xs2.reshape((2500,1))),axis=1), dtype=torch.float32)
targets = torch.tensor(Ys.reshape((2500,1)), dtype=torch.float32)


#  function is not linear and has multiple arguments,
#  let us create a model with non-linear activation function
#  model definition

N_input = 2
N_hidden = 25
N_output = 1

#we will now have more hidden layers and more neurons in each of them

model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden, N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_output),
    nn.ReLU()
)

criterion = torch.nn.MSELoss() #  Mean Square Error

#  Construct the optimizer - Now we use Adam, a bit smarter optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epoch = 50000
loss_val = []

#  Gradient Descent
for epoch in range(N_epoch):
    #  Forward pas: Compute predicted y passing x to the model
    y_pred = model(inputs)

    loss = criterion(y_pred,targets)
    if epoch % 2000 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())
        loss_val.append(loss.item())

    #  Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    #  perform a backward pass (backpropagation)
    loss.backward()
    #  Update the parameters
    optimizer.step()

plt.plot(range(len(loss_val)),loss_val,label='loss over epochs')
plt.legend()
plt.show()


test_Xs1 = np.linspace(-20,20,15)
test_Xs2 = np.linspace(-20,20,15)

test_Xs1, test_Xs2 = np.meshgrid(test_Xs1,test_Xs2)
a=1
with torch.no_grad():
    test_data = torch.tensor(np.concatenate((test_Xs1.reshape((225,1)),test_Xs2.reshape((225,1))),axis=1),dtype=torch.float32)
    test_output = model(test_data)

test_mesh_output = test_output.reshape((15,15))
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(Xs1,Xs2,Ys,label='training data')
ax.scatter(test_Xs1,test_Xs2,test_mesh_output,label='testing data')

e=1