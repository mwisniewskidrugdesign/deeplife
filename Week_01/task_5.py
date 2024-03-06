import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#  Overfitting Noise with Large Capacity Network

# we create the training data
train_noisy_Xs=np.random.rand(20)*10
train_noisy_Xs.sort()
train_noisy_Ys=train_noisy_Xs**2-5+np.random.rand(20)*40-20 #20 is the average of the noise

# and the validation data
val_noisy_Xs=np.random.rand(20)*10
val_noisy_Xs.sort()
val_noisy_Ys=val_noisy_Xs**2-5+np.random.rand(20)*40-20

#  plt.plot(train_noisy_Xs,train_noisy_Ys,label="training data")
#  plt.plot(val_noisy_Xs,val_noisy_Ys,label="validation data")
#  plt.legend()

train_inputs = torch.tensor(train_noisy_Xs.reshape((20,1)),dtype=torch.float32)
train_targets = torch.tensor(train_noisy_Ys.reshape((20,1)),dtype=torch.float32)

val_inputs = torch.tensor(val_noisy_Xs.reshape((20,1)),dtype=torch.float32)
val_targets = torch.tensor(val_noisy_Ys.reshape((20,1)),dtype=torch.float32)

#we set the parameters of our relatively complex network architecture
N_input = 1
N_hidden = 500
N_output = 1

#  we have few datapoints so we are making big capacity
model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_output)
)
criterion = torch.nn.MSELoss()  # Mean Square Error
# Construct the optimizer (Now we use Adam, a bit smarter optimizer)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epochs = 10000
train_loss_vals = []
val_loss_vals = []

for epoch in range(N_epochs):
    train_y_pred = model(train_inputs)
    train_loss = criterion(train_y_pred,train_targets)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        train_loss_vals.append(train_loss.item())
        with torch.no_grad():
            val_y_pred = model(val_inputs)
            val_loss = criterion(val_y_pred,val_targets)
            val_loss_vals.append(val_loss)
        print('epoch: ',epoch,', loss: ',train_loss.item(),', valid loss: ',val_loss.item())

#  plt.plot(train_loss_vals,label="training loss")
#  plt.plot(val_loss_vals,label="validation loss")
#  plt.legend()

#  plt.figure()

#  plt.plot(train_noisy_Xs,train_noisy_Ys,label="training data")
#  plt.plot(val_noisy_Xs,val_noisy_Ys,"r-",label="validation data")
#  plt.plot(train_inputs,train_y_pred.detach().numpy(),"bx",label="training output")
#  plt.plot(val_inputs,val_y_pred.detach().numpy(),"rx",label="validation output")

#  actual_Xs=np.linspace(0,10,100)
#  plt.plot(actual_Xs,actual_Xs**2-5,"g:",label="actual function")
#  plt.legend()



#  THIS MODEL IS TOO COMPLICATED SO IT IS MAKING OVERFITTING OR MAYBE WE HAVE NOT ENOUGH DATA

#  let us createlarger training data

train_noisy_Xs=np.random.rand(1000)*10
train_noisy_Xs.sort()
train_noisy_Ys=train_noisy_Xs**2-5+np.random.rand(1000)*40-20 #20 is the average of the noise

# and the validation data
val_noisy_Xs=np.random.rand(200)*10
val_noisy_Xs.sort()
val_noisy_Ys=val_noisy_Xs**2-5+np.random.rand(200)*40-20

#plot what is generated

#  plt.plot(train_noisy_Xs,train_noisy_Ys,"r:",label="training data")
#  plt.plot(val_noisy_Xs,val_noisy_Ys,"g-",label="validation data")
#  plt.legend()

#  convert our numpy arrays Xs and Ys into torch vectors we can use in training

train_inputs = torch.tensor(train_noisy_Xs.reshape((1000,1)), dtype=torch.float32)
train_targets = torch.tensor(train_noisy_Ys.reshape((1000,1)), dtype=torch.float32)

val_inputs=torch.tensor(val_noisy_Xs.reshape((200,1)), dtype=torch.float32)
val_targets=torch.tensor(val_noisy_Ys.reshape((200,1)), dtype=torch.float32)

N_input = 1
N_hidden = 50
N_output = 1

model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_output)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epochs = 10000
train_loss_vals = []
val_loss_vals = []

for epoch in range(N_epochs):
    train_y_pred = model(train_inputs)
    train_loss = criterion(train_y_pred,train_targets)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        train_loss_vals.append(train_loss.item())
        with torch.no_grad():
            val_y_pred = model(val_inputs)
            val_loss = criterion(val_y_pred,val_targets)
            val_loss_vals.append(val_loss.item())
        print('epoch: ',epoch,', loss: ',train_loss.item(),', val_loss: ',val_loss.item())

plt.plot(train_loss_vals,label="training loss")
plt.plot(val_loss_vals,label="validation loss")
plt.legend()

plt.figure()

plt.plot(train_noisy_Xs,train_noisy_Ys,label="training data")
plt.plot(val_noisy_Xs,val_noisy_Ys,"r-",label="validation data")
plt.plot(train_inputs,train_y_pred.detach().numpy(),"bx",label="training output")
plt.plot(val_inputs,val_y_pred.detach().numpy(),"rx",label="validation output")

actual_Xs=np.linspace(0,10,2000)
plt.plot(actual_Xs,actual_Xs**2-5,"g:",label="actual function")
plt.legend()
plt.show()