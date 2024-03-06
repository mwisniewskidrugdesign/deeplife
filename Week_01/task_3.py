import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# let us start with some random X1,X2 point pairs from 0,2 interval
N_samples=100
train_X1s=np.random.rand(N_samples)*2
train_X2s=np.random.rand(N_samples)*2

def XOR_like(x1,x2):
    if (x1>=1 and x2<=1) or (x2>=1 and x1<=1):
        return 1
    else:
        return 0
train_Ys=np.array([XOR_like(x1,x2) for x1,x2 in zip(train_X1s,train_X2s)])

#prepare input and output data
inputs = torch.tensor(np.concatenate((train_X1s.reshape((N_samples,1)),train_X2s.reshape((N_samples,1))),axis=1), dtype=torch.float32)
targets = torch.tensor(train_Ys.reshape((N_samples,1)), dtype=torch.float32)


#  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#  ax.scatter(train_X1s, train_X2s, train_Ys,label="training data",vmin=-1,c=train_Ys) #data is colored by the Y value

N_input = 2
N_hidden = 20
N_output = 1

model = nn.Sequential(
    nn.Linear(N_input,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden,N_output),
    nn.ReLU()
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epochs = 15000
loss_vals = []

for epoch in range(N_epochs):
    y_pred = model(inputs)
    loss = criterion(y_pred,targets)
    if epoch % 500 == 0:
        loss_vals.append(loss.item())
        print('epoch: ',epoch,', loss: ',loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

N_samples = 100
test_X1s=np.random.rand(N_samples)*2
test_X2s=np.random.rand(N_samples)*2
test_Ys = np.array([XOR_like(x1,x2) for x1,x2 in zip(test_X1s,test_X2s)])

test_inputs = torch.tensor(np.concatenate((
    test_X1s.reshape((N_samples,1)),
    test_X2s.reshape((N_samples,1))),axis=1),dtype=torch.float32)

with torch.no_grad():
    test_output = model(test_inputs)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(train_X1s, train_X2s, train_Ys,label="training data",c=train_Ys,marker=".")
ax.scatter(test_X1s,test_X2s,test_output,label="testing data",c=test_output,marker="o")
plt.show()