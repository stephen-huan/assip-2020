import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from frqParse import train, validation, test

parser = get_params_parser()
params = parser.parse_args()


x=torch.from_numpy(append(train))
y=torch.from_numpy(scores(train))

x_test=torch.from_numpy(append(test))
y_test=torch.from_numpy(validation(test))

x_valid=torch.from_numpy((append(validation))
y_valid=torch.from_numpy((scores(validation))
H=500 #hidden dimension

model = torch.nn.Sequential(
    torch.nn.Linear(append(train)[0].size, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, scores(train)[0].size),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):          #training
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

def append(df):
    input=[]
    index=0

    input[index]=embed(df)[index].append(one_hot_encoder(df)[index])

def scores(df):
    return df["score"].tolist()

def one_hot_encoder(df):
    encoded=[]
    index=0

    for set in df["essay_set"]:
        v = np.zeros(9)
        v[essay_set - 1] = 1
        encoded[index+=1]=v

    return encoded

def embed(df):
    embed=[]
    index=0

    for response in df["essay"]:
        embed[index+=1]=make_batch(response)

    return embed

def get_params_parser():

    parser = argparse.ArgumentParser(description='ASAG Classifier')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--outputdir", type=str, required=True, help="Path to save model")

    return parser
