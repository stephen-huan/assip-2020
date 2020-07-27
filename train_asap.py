import sys, os, argparse, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import data
from data import _train, _validation, _test, train, validation, test

learning_rate = 1e-4

def get_model(sizes: list=[300]) -> nn.Sequential:
    """ Defines a torch model. """
    sizes = [809] + sizes + [data.MAX_SCORE + 1]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # use softmax for last layer, intermediate layers ReLU
        layers.append(nn.Softmax(dim=1) if i == len(sizes) - 2 else nn.ReLU())
    return nn.Sequential(*layers)

def train_test_split(dataset: TensorDataset, split: float) -> tuple: 
    """ Splits a dataset into train and test splits. """
    train_size = int(split*len(dataset))
    return random_split(dataset, (train_size, len(dataset) - train_size))

def make_dataset(X: np.array, y: np.array) -> DataLoader:
    """ Turns a numpy feature vector and labels into a pytorch DataLoader. """
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y)) 
    
def make_dataloader(dataset: TensorDataset) -> DataLoader:
    """ Makes a DataLoader from a dataset. """ 
    return DataLoader(dataset, shuffle=True, 
                      batch_size=args.batch_size, 
                      num_workers=args.num_workers)

def un_encode(v: torch.Tensor) -> torch.Tensor:
    """ Undo one hot encoding. """
    return v.argmax(axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASAG Classifier")
    parser.add_argument("-o", "--outputdir", required=True, 
                        help="Path to save model.")
    parser.add_argument("-l", "--load_model", default=None, 
                        help="Path to a pretrained model file to load for additional training.")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-p", "--split", type=float, default=0.9, 
                        help="Fraction of the training set to withhold for validation.")
    parser.add_argument("-s", "--seed", type=int, default=1234, 
                        help="Random seed")
    parser.add_argument("-n", "--num_workers", type=int, default=5)
    parser.add_argument("-f", "--validation_frequency", type=int, default=100, 
                        help="Number of mini-batches until running on validation.")
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)

    # data
    dataset = make_dataset(*train)
    train_dataset, validation_dataset = train_test_split(dataset, args.split)
    train_generator = make_dataloader(train_dataset)
    valX, valy = map(lambda x: torch.stack(x), zip(*validation_dataset))

    model = get_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train on multiple GPUs if possible, defaulting to CPU
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")
    print(f"using {torch.cuda.device_count()} GPUs for training")
    if CUDA:
        model = nn.DataParallel(model)
    model.to(device)
    valX, valy = valX.to(device), valy.to(device)

    min_val = float("inf")
    for epoch in range(args.epochs):
        # training
        total_loss = batches = 0
        start = time.time()
        for batch, (batchX, batchy) in enumerate(train_generator):
            batchX, batchy = batchX.to(device), batchy.to(device)
            # forward
            y_pred = model(batchX)
            loss = loss_fn(y_pred, batchy)
            
            total_loss += loss.item()
            batches += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if batches == args.validation_frequency:
                batches = 0
                
                with torch.no_grad():
                    y_pred = model(valX)
                    val_loss = loss_fn(y_pred, valy)
                    correct = (torch.argmax(y_pred, 1) == valy).sum().item()

                # save best model
                if val_loss < min_val:
                    min_val = val_loss
                    torch.save(model, f"{args.outputdir}/model.net") 

                # summary
                print(f"epoch: {epoch}, batch: {batch}, loss: {total_loss/args.validation_frequency}")
                print(f"validation loss: {val_loss}, validation accuracy: {correct/len(valX)}")
                print(f"took {time.time() - start} seconds for {args.validation_frequency} batches")
                total_loss = 0
                start = time.time()
    
    if min_val < float("inf"):
        print("loading best model:")
        model = torch.load(f"{args.outputdir}/model.net")
        if not isinstance(model, nn.Sequential):
            model = model.module

    with torch.no_grad():
        y_pred = model(valX)
        val_loss = loss_fn(y_pred, valy)
        correct = (torch.argmax(y_pred, 1) == valy).sum().item()

    print(f"validation loss: {val_loss}, validation accuracy: {correct/len(valX)}")



