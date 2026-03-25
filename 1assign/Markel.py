from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np


df=pd.read_csv("./1assign/data/insurance.csv")

#DATA CLEANING
df["smoker"]=[0 if elem=="no" else 1 for elem in df["smoker"]]

df=pd.get_dummies(df,dtype=int)
df=df.drop(columns="sex_male")


X=df.drop(columns="charges")
y=df["charges"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  

X_test = scaler.transform(X_test)  

X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)
y_train=torch.tensor(y_train.values, dtype=torch.float32)
y_test=torch.tensor(y_test.values, dtype=torch.float32)

def min_square_error(v_true,v_pred):
    difs=(v_true-v_pred)**2

    return np.mean(difs)

class RegressionMarkel(nn.Module):
    def __init__(self):
        super().__init__()

        self.capa1=nn.Linear(9,32)
        self.capa2=nn.Linear(32,1)
        


        self.relu = nn.ReLU()


    def forward(self,x):
        x=self.capa1(x)
        x=self.relu(x)
        x=self.capa2(x)
        
        return x

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(f"runs/insurance_{timestamp}")

#TRAIN
model=RegressionMarkel()
loss_func=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.2)


model.train()

epochs=1000
for epoch in range(epochs):
    predictions=model(X_train).squeeze()
    # print(predictions)
    # print("_________")
    # print(y_train)
    loss=loss_func(predictions,y_train)
    
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # actualizar pesos
    optimizer.step()

    writer.add_scalar("Loss/train", loss.item(), epoch)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

#TEST
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    y_test=y_test.numpy()
    predictions=predictions.numpy()
    #print(predictions)
    print(min_square_error(y_test,predictions))





