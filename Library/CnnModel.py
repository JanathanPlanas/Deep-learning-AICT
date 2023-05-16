
# Make device agnostic code
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Standard PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optimpip
from helper_functions import accuracy_fn
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import ConfusionMatrix
# Import tqdm for progress bar
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TorchDataset():

    def __init__(self):

        pass

    def Spliting(self,data,random_state, test_size , shuffle: bool) :


            X = data[:, :-1]
            y = data[:, -1]
                            
            X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                                test_size=test_size, 
                                                                random_state=random_state, shuffle = shuffle)
        
            
            self.X_train= torch.tensor(X_train)
            self.X_test= torch.tensor(X_test)
            self.y_train= torch.tensor(y_train)
            self.y_test= torch.tensor(y_test)
            
            return X_train, X_test, y_train, y_test

            # convertendo numpy arrays em tensores do PyTorch
    def DataLoaders(self,batch_size):

            # criando datasets
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test , self.y_test)

            # criando dataloaders 
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
                    

        return  train_dataloader,  test_dataloader
    

# carregando o modelo

   
    def train_step(self, model: torch.nn.Module,
                data_loader_train: torch.utils.data.DataLoader,
                data_loader_test: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device ,
                epochs:int ):
        # loop pelo dataloader de treino

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n---------")
            # loop pelo dataloader de treino
            model.to(device)
            model.train()
            training_loss= 0
            training_accurary = 0
            valid_loss = 0
            for batch, (inputs, target) in enumerate(data_loader_train):
                # movendo os dados para o dispositivo de processamento
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(2)
                # fazendo as previsões
                output = model(inputs)
                
                # calculando a perda
                loss = loss_fn(output, target.long())

                training_loss +=loss.data.item()
                training_accurary += accuracy_fn(target, output.argmax(dim=1))
                
                # retropropagando os gradientes e atualizando os pesos
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                

            training_loss/= len(data_loader_train)
            training_accurary /= len(data_loader_train)
                
                # imprimindo as métricas de treino a cada 50 lotes
                
            print(f'Train loss: {training_loss:.5f} | Train accuracy: {training_accurary:.2f}%')
                
        # avaliando o modelo no dataloard de teste
                # loop pelo dataloader de teste

            model.eval()
            valid_loss = 0
            test_loss = 0
            test_accurary = 0
            with torch.inference_mode():
                
                for data, target in data_loader_test:
                    # movendo os dados para o dispositivo de processamento
                    data = data.to(device)
                    target = target.to(device)
                    data = data.unsqueeze(2)
                    
                    # 1. Forward pass
                    test_pred = model(data)
                    # 2. Calculate loss and accuracy
                    test_loss +=loss_fn(test_pred, target.long())
                    valid_loss += test_loss.data.item()
                    test_accurary += accuracy_fn(target, test_pred.argmax(dim=1))
                    
                valid_loss /= len(data_loader_test)
                test_accurary /= len(data_loader_test)
                    
            print(f'Test loss: {valid_loss:.5f} | Test accuracy: {test_accurary:.2f}%')


    def print_train_time(start: float, end: float, device: torch.device = device):
        """Prints difference between start and end time.

        Args:
            start (float): Start time of computation (preferred in timeit format). 
            end (float): End time of computation.
            device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
            float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time


    def Making_Predictions(self, data_loader : torch.utils.data.DataLoader,
                           model: torch.nn.Module):
        # Import tqdm for progress bar

        # 1. Make predictions with trained model
        y_preds = []
        model.eval()
        with torch.inference_mode():
            for X, y in tqdm(data_loader, desc="Making predictions"):
                # Send data and targets to target device
                X, y = X.to(device), y.to(device)

                X = X.unsqueeze(2)
                # Do the forward pass
                y_logit = model(X)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())
            # Concatenate list of predictions into a tensor
            y_pred_tensor = torch.cat(y_preds)

            
        return  y_pred_tensor
    
    def print_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes:int):

        class_names = ['CLEAR','WIFI','LTE']
        confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        confmat_tensor = confmat(preds=  y_pred,
                                target= y_true )

        # 3. Plot the confusion matrix  
        plt.figure()
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
            class_names=class_names, # turn the row and column labels into class names
            figsize=(10, 7) )




class CNNModel(nn.Module):

    def __init__(self,in_channels): 
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=512, 
                      kernel_size=2, 
                      padding=1, stride = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, 
                      out_channels=32, 
                      kernel_size=2, 
                      padding=1,stride = 1), 
                      nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2 ) )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      padding=1,stride = 1),
             nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Conv1d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=2, 
                      padding=1,stride = 1),
                      nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2) )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        stride = 1),
                        nn.BatchNorm1d(256),
                        nn.Tanh(),
                        nn.Conv1d(in_channels=256,
                        out_channels=256,
                        kernel_size=2,
                        padding=1,
                        stride = 1),
                        nn.BatchNorm1d(256),
                        nn.Tanh(),
                        nn.MaxPool1d(kernel_size=2) )

        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4),
            nn.LogSoftmax(dim=1)

            )
        
 
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        out = nn.functional.softmax(out, dim=1)
        
        return out
    




    

