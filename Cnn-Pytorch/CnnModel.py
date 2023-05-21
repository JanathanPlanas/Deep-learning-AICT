
# Make device agnostic code
from timeit import default_timer as timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
# Standard PyTorch importspip
import torch
import torch.nn as nn
from Data_Loader import DATA_1M
from helper_functions import accuracy_fn
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from torchmetrics import ConfusionMatrix
# Import tqdm for progress bar
import pandas as pd 
from dataclasses import dataclass
from tqdm.auto import tqdm


class NeuralNetCNN():

    def __init__(self, columns):

        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        
        self.Cnn = Classifier(in_channels_columns= columns)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.Cnn.parameters(), lr=0.01)

    def __str__(self) -> str:
        
        return f"Device on {self.device} \n Model {self.Cnn}"
    
    def __setitem__ (self, value):

        self.loss_fn = value
        


# carregando o modelo
    
   
    def training_loop(self, model: torch.nn.Module,
                data_loader_train: torch.utils.data.DataLoader,
                data_loader_test: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device ,
                epochs:int, inplace = False ):
        # loop pelo dataloader de treino

        results_array_test = np.empty((epochs, 2))
        results_array_train = np.empty([epochs, 2])
        
        print(f"Training on {self.device}")
        for epoch in tqdm(range(epochs)):
            print(f" Epoch: {epoch}\n---------")
            # loop pelo dataloader de treino
            model.to(device)
            model.train().double()
            training_loss= 0
            training_accurary = 0
            valid_loss = 0
            for batch, (inputs, target) in enumerate(data_loader_train):
                # movendo os dados para o dispositivo de processamento
                inputs = inputs.to(device).double()
                inputs = inputs.unsqueeze(2 )
                # fazendo as previsões
                output = model(inputs.double())
                
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

            results_array_train[epoch, 0] = training_loss
            results_array_train[epoch, 1] = training_accurary
            
                
        # avaliando o modelo no dataloard de teste
                # loop pelo dataloader de teste

            model.eval().double()
            valid_loss = 0
            test_loss = 0
            test_accurary = 0
            accuracy_list = []
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

            results_array_test[:, 0] = valid_loss
            results_array_test[:, 1] = test_accurary

        self.test_acc = results_array_test[:, 1]
        self.tess_loss = results_array_test[:, 0]

        self.train_acc = results_array_train[:, 1]
        self.train_loss = results_array_train[:, 0]

        if inplace == True:
            return self.test_acc , self.train_cc, self.tess_loss, self.train_loss
        else:
            pass

    def __call__(self, train: bool, test : bool) -> Any:

            if train == True and test == True:

                return pd.DataFrame({
                    "Test Accuracy": self.test_acc , 
                    "Test Loss ":self.tess_loss,
                    "Train Accuracy": self.train_acc,
                    "Train Loss": self.train_loss 
                    })
            if  test == True  and train == False:
                return pd.DataFrame({
                    "Test Accuracy": self.test_acc ,
                    "Test Loss ":self.tess_loss
                })
            
            if  train == True and test == False :
                return pd.DataFrame({
                    "Train Accuracy": self.train_acc ,
                    "Train Loss ":self.train_loss
                })


    def print_train_time(start: float, end: float, device: torch.device = 'cpu'):
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
            for X, y in (data_loader):
                # Send data and targets to target device
                X, y = X.to(self.device), y.to(self.device)

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
        plt.show()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = 2,padding= 1, stride = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size = 2,padding= 1, stride = 1),
            nn.BatchNorm1d(out_channels),
             nn.Tanh(),
             nn.MaxPool1d(kernel_size=2),
        )

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_channels_columns ):
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels_columns, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            # ConvBlock(in_channels=128, out_channels=256),
            # ConvBlock(in_channels=256, out_channels=512),
        )
        
        self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
#             nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):

        out = self.conv(x)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        out = nn.functional.softmax(out, dim=1)
        return out



class Putting_All_Together():

    def Running(self):

        Data_loader = DATA_1M(seconds=5,columns=2000, jump_time =10, n_jumps=1)
        Torch = NeuralNetCNN(columns= Data_loader(Fourier=True).shape[1] -1)

        Data_loader.Spliting(data= Data_loader(Fourier=True), random_state= 42, test_size = 0.275, shuffle = True, inplace= False)
        train_dataloader , test_dataloader = Data_loader.DataLoaders(batch_size=256, inplace=True)

        Torch.training_loop(data_loader_train=train_dataloader,
        data_loader_test = test_dataloader,
                model=Torch.modelCnnModel, 
                loss_fn=Torch.loss_fn,
                optimizer=Torch.optimizer,
                accuracy_fn=accuracy_fn,
                device=Torch.device,
                epochs = 5)
        
        print(Torch(test= True, train= True))

        

        class_names = ['CLEAR','WIFI','LTE']
        # confmat = ConfusionMatrix(num_classes=3, task='multiclass')
        # confmat_tensor = confmat(preds=  Torch.Making_Predictions(model = Torch.modelCnnModel, data_loader= Data_loader.test_dataloader),
        #                         target= Data_loader.y_test )
        
        print(classification_report(Data_loader.y_test, Torch.Making_Predictions(model = Torch.modelCnnModel, data_loader= Data_loader.test_dataloader),target_names=class_names))

if __name__ =="__main__":

        data = DATA_1M(seconds=5,columns=2000, jump_time =10, n_jumps=1)
        Torch = NeuralNetCNN(columns= data(Fourier=True).shape[1] -1)

        data.Spliting(data= data(Fourier=True), random_state= 42, test_size = 0.275, shuffle = True, inplace= False)
        train_dataloader , test_dataloader = data.DataLoaders(batch_size=256, inplace=True)

        Torch.training_loop(data_loader_train=train_dataloader,
        data_loader_test = test_dataloader,
                model=Torch.Cnn, 
                loss_fn=Torch.loss_fn,
                optimizer=Torch.optimizer,
                accuracy_fn=accuracy_fn,
                device=Torch.device,
                epochs = 5)
        
        print(Torch(test= True, train= True))

        

        class_names = ['CLEAR','WIFI','LTE']
        # confmat = ConfusionMatrix(num_classes=3, task='multiclass')
        # confmat_tensor = confmat(preds=  Torch.Making_Predictions(model = Torch.modelCnnModel, data_loader= Data_loader.test_dataloader),
        #                         target= Data_loader.y_test )
        
        print(classification_report(data.y_test, Torch.Making_Predictions(model = Torch.Cnn, data_loader= test_dataloader),target_names=class_names))

    


# class CNNModel(nn.Module):

#     def __init__(self,in_channels): 
#         super(CNNModel, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, 
#                       out_channels=512, 
#                       kernel_size=2, 
#                       padding=1, stride = 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=512, 
#                       out_channels=32, 
#                       kernel_size=2, 
#                       padding=1,stride = 1), 
#                       nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2 ) )
        
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(in_channels=32, 
#                       out_channels=64, 
#                       kernel_size=3, 
#                       padding=1,stride = 1),
#              nn.BatchNorm1d(64),
#             nn.Tanh(),
#             nn.Conv1d(in_channels=64, 
#                       out_channels=128, 
#                       kernel_size=2, 
#                       padding=1,stride = 1),
#                       nn.BatchNorm1d(128),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2) )

#         self.layer3 = nn.Sequential(
#                         nn.Conv1d(in_channels=128,
#                         out_channels=256,
#                         kernel_size=3,
#                         padding=1,
#                         stride = 1),
#                         nn.BatchNorm1d(256),
#                         nn.Tanh(),
#                         nn.Conv1d(in_channels=256,
#                         out_channels=128,
#                         kernel_size=2,
#                         padding=1,
#                         stride = 1),
#                         nn.BatchNorm1d(256),
#                         nn.Tanh(),
#                         nn.MaxPool1d(kernel_size=2) )
        
#         self.layer4 = nn.Sequential(
#                         nn.Conv1d(in_channels=128,
#                         out_channels=256,
#                         kernel_size=3,
#                         padding=1,
#                         stride = 1),
#                         nn.BatchNorm1d(256),
#                         nn.Tanh(),
#                         nn.Conv1d(in_channels=256,
#                         out_channels=256,
#                         kernel_size=2,
#                         padding=1,
#                         stride = 1),
#                         nn.BatchNorm1d(256),
#                         nn.Tanh(),
#                         nn.MaxPool1d(kernel_size=2) )

        
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=256, out_features=128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=4),
#             nn.LogSoftmax(dim=1)

#             )
 
        
    # def forward(self, x):
        
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)

    #     out = out.view(out.size(0), -1)
    #     out = self.classifier(out)
        
    #     out = nn.functional.softmax(out, dim=1)
        
    #     return out