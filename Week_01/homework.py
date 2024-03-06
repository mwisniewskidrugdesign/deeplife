import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, Xs1, Xs2, transform=None):
        # Konstruktor klasy CustomDataset
        self.Xs1,self.Xs2 = self.convert_inputs(Xs1,Xs2)
        self.transform = transform
        self.data = self.generate_data()
        self.targets = self.generate_targets()
    def __len__(self):
        # Funkcja zwracająca ilość próbek w zestawie danych
        return len(self.data)
    def __getitem__(self, idx):
        # Funkcja zwracająca próbkę o zadanym indeksie
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def convert_inputs(self,Xs1,Xs2):
        N_Xs1 = len(Xs1)
        N_Xs2 = len(Xs2)
        Xs1, Xs2 = np.meshgrid(Xs1, Xs2)
        return Xs1.reshape((N_Xs1*N_Xs2,1)), Xs2.reshape((N_Xs1*N_Xs2,1))

    def generate_data(self):
        # Funkcja generująca wszystkie możliwe kombinacje X1 i X2 z Xs1 i Xs2
        data = np.concatenate((self.Xs1,self.Xs2),axis=1)
        return torch.tensor(data, dtype=torch.float32)
    def generate_targets(self):
        # Funkcja generująca Ys dla wszystkich mozliwych kominacji inputów
        return torch.tensor((self.Xs2**2 * 5) - (self.Xs1 * 3) + 15, dtype=torch.float32)


# Dane Xs1 i Xs2
Xs1 = np.linspace(-10, 10, 50)
Xs2 = np.linspace(-10, 10, 50)

dataset = CustomDataset(Xs1, Xs2)
dataloader = DataLoader(dataset,batch_size=640,shuffle=True)
a=1

N_input = 2
N_hidden = 25
N_output = 1

# Tworzenie instancji modelu

model = nn.Sequential(
    nn.Linear(N_input, N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden, N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden, N_hidden),
    nn.ReLU(),
    nn.Linear(N_hidden, N_output),
    nn.ReLU()
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

N_epoch = 50000
loss_val = []
for epoch in range(N_epoch):
    for batch_idx,data in enumerate(dataloader):
        inputs = data
        targets = dataset.targets[batch_idx*640:(batch_idx+1)*640]
        y_pred = model(inputs)
        optimizer.zero_grad()
        loss = criterion(y_pred,targets)

        loss.backward()
        optimizer.step()
    a=1
    if epoch % 2000 == 0:
        print('epoch: ',epoch,', loss: ',loss.item())
