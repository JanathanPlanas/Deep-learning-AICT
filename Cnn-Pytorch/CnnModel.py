# Make device agnostic code
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
# Import tqdm for progress bar
import pandas as pd
# Standard PyTorch importspip
import torch
import torch.nn as nn
import torch.optim as optim
from Data_Loader import DATA_1M
from helper_functions import accuracy_fn
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm


class NeuralNetCNN():
    """
A classe NeuralNetCNN representa um modelo de rede neural convolucional.

Construtor:

__init__(self, colunas): Inicializa a classe.
colunas (int): O número de canais de entrada.
Atributos:

dispositivo (torch.device): O dispositivo usado para computação (CUDA, se disponível, caso contrário, CPU).
Cnn (Classificador): O modelo CNN.
ExtenderClassifier (ExtenderClassifier): O modelo de classificador estendido.
loss_fn (nn.CrossEntropyLoss): A função de perda usada para treinamento.
otimizador (torch.optim.Adam): O otimizador para atualizar os parâmetros do modelo CNN.
optimizer_extended (torch.optim.Adam): O otimizador para atualizar os parâmetros do modelo de classificador estendido.
Métodos:
    
__str__(self) -> str: Retorna uma representação de string da classe, incluindo o dispositivo e o modelo CNN.
__setitem__(self, value): Define a função de perda para o valor especificado.
Observação: o trecho de código fornecido está incompleto e o objetivo do método __setitem__ não é claro sem um contexto adicional.  

    """

    def __init__(self, columns, conv_blocks, groupblocks):

        self.columns_shape = columns

        self.device = (torch.device('cuda') if torch.cuda.is_available()
                       else torch.device('cpu'))

        self.Cnn = Classifier(in_channels_columns=columns, conv_blocks =  conv_blocks , group_blocks=groupblocks )

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.Cnn.parameters(), lr= 1e-3)

    def __str__(self) -> str:

        return f"Device on {self.device} \n Model {self.Cnn}"

    def __setitem__(self, value):

        self.loss_fn = value


   
    def count_blocks(self, model:torch.nn.Module):
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                count += 1
        return count

    # carregando o modelo e fazendo automatica os prints do treino e teste de acurácia de loss functions

    def training_loop(self, model: torch.nn.Module,
                      data_loader_train: torch.utils.data.DataLoader,
                      data_loader_test: torch.utils.data.DataLoader,
                      loss_fn: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      accuracy_fn,
                      device: torch.device,
                      epochs: int, 
                      writer: torch.utils.tensorboard.writer.SummaryWriter
                      ,inplace=False):
        """Treina e testa um modelo PyTorch.

     Passa um modelo PyTorch de destino por meio de train_step() e test_step()
     funções para um número de épocas, treinando e testando o modelo
     no mesmo loop de época.

     Calcula, imprime e armazena métricas de avaliação.

     Argumentos:
       model: um modelo PyTorch a ser treinado e testado.
       train_dataloader: Uma instância do DataLoader para o modelo a ser treinado.
       test_dataloader: Uma instância do DataLoader para o modelo a ser testado.
       otimizador: Um otimizador PyTorch para ajudar a minimizar a função de perda.
       loss_fn: uma função de perda do PyTorch para calcular a perda em ambos os conjuntos de dados.
       epochs: Um número inteiro indicando para quantas épocas treinar.
       dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").
      
     Retorna:
       Um dicionário de perda de treinamento e teste, bem como treinamento e
       testar métricas de precisão. Cada métrica tem um valor em uma lista para
       cada época.
       Na forma: {train_loss: [...],
                 train_acc: [...],
                 teste_perda: [...],
                 test_acc: [...]}
       Por exemplo, se o treinamento for epochs=2:
               {train_loss: [2.0616, 1.0537],
                 train_acc: [0,3945, 0,3945],
                 perda_teste: [1.2641, 1.5706],
                 test_acc: [0,3400, 0,2973]}
     """

        results_array_test = np.empty((epochs, 2))
        results_array_train = np.empty([epochs, 2])

        print(f"Training on {self.device}")
        for epoch in tqdm(range(epochs)):
            print(f" Epoch: {epoch}\n---------")
            # loop pelo dataloader de treino
            model.to(device)
            model.train().double()
            training_loss = 0
            training_accurary = 0
            valid_loss = 0
            for batch, (inputs, target) in enumerate(data_loader_train):
                # movendo os dados para o dispositivo de processamento
                inputs = inputs.to(device).double()
                inputs = inputs.unsqueeze(2)
                
                # fazendo as previsões
                output = model(inputs.double())

                # calculando a perda
                loss = loss_fn(output, target.long())

                training_loss += loss.data.item()
                training_accurary += accuracy_fn(target, output.argmax(dim=1))

                # retropropagando os gradientes e atualizando os pesos
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            training_loss /= len(data_loader_train)
            training_accurary /= len(data_loader_train)

            # imprimindo as métricas de treino a cada 50 lotes

            print(
                f'Train loss: {training_loss:.5f} | Train accuracy: {training_accurary:.2f}%')

            results_array_train[epoch, 0] = "%.2f" % training_loss
            results_array_train[epoch, 1] = "%.2f" % training_accurary

            # avaliando o modelo no dataloard de teste
            # loop pelo dataloader de teste

            model.eval().double()
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
                    test_loss += loss_fn(test_pred, target.long())
                    valid_loss += test_loss.data.item()
                    test_accurary += accuracy_fn(target,
                                                 test_pred.argmax(dim=1))

                valid_loss /= len(data_loader_test)
                test_accurary /= len(data_loader_test)

            print(
                f'Test loss: {valid_loss:.5f} | Test accuracy: {test_accurary:.2f}%')

            results_array_test[epoch, 0] = "%.2f" % valid_loss
            results_array_test[epoch, 1] = "%.2f" % test_accurary
            
            
            if writer:
                writer.add_scalars(main_tag=f"Loss per epochs _{len(list(self.Cnn.children()))} Layers",
                                tag_scalar_dict={"Test Loss": valid_loss},
                                global_step=epoch)

                # Add accuracy results to SummaryWriter
                writer.add_scalars(main_tag=f"Accuracy per epochs _{len(list(self.Cnn.children()))} Layers",
                                tag_scalar_dict={"Test Accuracy": test_accurary},
                                global_step=epoch)
                

                # Track the PyTorch model architecture
                writer.add_graph(model=model,
                                # Pass in an example input
                                input_to_model=torch.randn(8, self.columns_shape , 1).to(device).double())
                

                # Close the writer
                writer.close()

        self.test_acc = results_array_test[:, 1]
        self.tess_loss = results_array_test[:, 0]

        self.train_acc = results_array_train[:, 1]
        self.train_loss = results_array_train[:, 0]

        if inplace == True:
            return self.test_acc, self.train_cc, self.tess_loss, self.train_loss
        else:
            pass

    def __call__(self, train: bool, test: bool) -> Any:

        """
     Recupera as métricas de desempenho para treinamento e/ou teste.

     Argumentos:
         train (bool): sinalizador indicando se deve incluir métricas de treinamento.
         test (bool): sinalizador indicando se deve incluir métricas de teste.

     Retorna:
         pd.DataFrame ou None: DataFrame contendo as métricas solicitadas. Se train e test forem True,
                               o DataFrame incluirá métricas de treinamento e teste. Se apenas test for True,
                               ele incluirá apenas métricas de teste. Se apenas train for True, ele incluirá apenas
                               métricas de treinamento. Se train e test forem False, retorna None.

     """

        if train == True and test == True:
            return pd.DataFrame({
                "Test Accuracy": self.test_acc,
                "Test Loss ": self.tess_loss,
                "Train Accuracy": self.train_acc,
                "Train Loss": self.train_loss
            })
        if test == True and train == False:
            return pd.DataFrame({
                "Test Accuracy": self.test_acc,
                "Test Loss ": self.tess_loss
            })

        if train == True and test == False:
            return pd.DataFrame({
                "Train Accuracy": self.train_acc,
                "Train Loss ": self.train_loss
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

    def Making_Predictions(self, data_loader: torch.utils.data.DataLoader,
                           model: torch.nn.Module):
        """
     Faz previsões usando um modelo treinado no carregador de dados fornecido.

     Argumentos:
         data_loader (torch.utils.data.DataLoader): DataLoader contendo os dados de entrada.
         modelo (torch.nn.Module): modelo treinado para ser usado para fazer previsões.

     Retorna:
         tocha.Tensor: Tensor contendo os rótulos previstos.

     """

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
            self.y_pred_tensor = torch.cat(y_preds)

        return self.y_pred_tensor

    def print_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):

        class_names = ['CLEAR', 'WIFI', 'LTE']
        confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        confmat_tensor = confmat(preds=y_pred,
                                 target=y_true)

        # 3. Plot the confusion matrix
        plt.figure()
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
            class_names=class_names,  # turn the row and column labels into class names
            figsize=(10, 7))
        plt.show()


class ConvBlock(nn.Module):
    """
    A classe ConvBlock representa um bloco de camadas convolucionais com normalização em lote e funções de ativação. Ele é projetado para processar dados de entrada unidimensionais.

Aqui está uma divisão da classe:

__init__(self, in_channels, out_channels): Inicializa a instância ConvBlock.

in_channels (int): Número de canais de entrada.
out_channels (int): Número de canais de saída.
Configura duas camadas convolucionais sequenciais (conv1 e conv2) com normalização em lote e funções de ativação.
forward(self, x): Executa a passagem direta pelo ConvBlock.

x (tocha.Tensor): Tensor de entrada.
Passa o tensor de entrada pelas camadas conv1 e conv2.
Retorna o tensor de saída.
A classe ConvBlock pode ser usada como um bloco de construção em uma rede neural convolucional para extrair recursos de dados de entrada unidimensionais.
    """
    def __init__(self, in_channels, out_channels, conv_blocks):
        super().__init__()
        # 1 layer com 4 subcamadas
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        if conv_blocks >= 2 : # 2 layer com 4 subcamadas

            self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        if conv_blocks >= 3 : # 3 layer com 4 subcamadas

            self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )


        if conv_blocks >= 4 : # 4 layer com 4 subcamadas

            self.conv4 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )
            
        if conv_blocks >= 5 : # 5 layer com 4 subcamadas

            self.conv5 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )
        if conv_blocks == 6 : # 5 layer com 4 subcamadas

            self.conv6 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=2, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )
            
    def forward(self,x):

        x = self.conv1(x)
        if hasattr(self, 'conv2'):
            x = self.conv2(x)
        if hasattr(self, 'conv3'):
            x = self.conv3(x)
        if hasattr(self, 'conv4'):
            x = self.conv4(x)
        if hasattr(self, 'conv5'):
            x = self.conv5(x)
        if hasattr(self, 'conv6'):
            x = self.conv6(x)
        return x

    @property
    def conv_blocks(self):
        return self._conv_blocks
    
    @conv_blocks.setter
    def conv_blocks(self, value):
        if value < 0 or value > 6:
            print("Erro: conv_blocks inserido maior do que 4. O valor padrão de 1 será usado.")
            self._conv_blocks = 1
        else:
            self._conv_blocks = value




class Classifier(ConvBlock):
    """
    A classe Classifier representa um modelo de rede neural convolucional (CNN) para classificação. Consiste em vários blocos convolucionais seguidos por uma camada classificadora totalmente conectada.

Aqui está uma divisão da classe:

__init__(self, in_channels_columns): Inicializa a instância do Classificador.

in_channels_columns (int): Número de canais/colunas de entrada.
Configura as camadas convolucionais usando a classe ConvBlock, com números crescentes de canais de saída.
Configura as camadas do classificador, incluindo normalização em lote, camadas lineares (totalmente conectadas) e funções de ativação.
forward(self, x): Executa a passagem direta pelo modelo Classificador.

x (tocha.Tensor): Tensor de entrada.
Passa o tensor de entrada pelas camadas convolucionais (conv) e remodela a saída.
Passa o tensor remodelado pelas camadas classificadoras (classificador).
Aplica uma função de ativação softmax ao tensor de saída e o retorna.
O modelo classificador é projetado para obter dados de entrada com vários canais/colunas, aplicar operações convolucionais para extrair recursos e, em seguida, classificar os recursos extraídos em uma das três classes usando as camadas totalmente conectadas e a ativação softmax.
    """
    def __init__(self, in_channels_columns, conv_blocks , group_blocks):
        super().__init__(in_channels=in_channels_columns, out_channels=32, conv_blocks=conv_blocks)


        if group_blocks == 1 : # 1 layer com 4 subcamadas
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns, out_channels=32, conv_blocks = conv_blocks),
            )

            self.classifier = nn.Sequential(
                #             nn.Dropout(0.2),
                nn.BatchNorm1d(32),
                nn.Linear(32, 20),
                nn.ReLU(),
                nn.BatchNorm1d(20),
                #             nn.Dropout(0.2),
                nn.Linear(20, 3),
                nn.LogSoftmax(dim=1)
            )

        if group_blocks == 2 : # 2 layer com 4 subcamadas
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns, out_channels=32, conv_blocks = conv_blocks),
                ConvBlock(in_channels=32, out_channels=48,conv_blocks = conv_blocks),
            )

            self.classifier = nn.Sequential(
                #             nn.Dropout(0.2),
                nn.BatchNorm1d(48),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                #             nn.Dropout(0.2),
                nn.Linear(32, 3),
                nn.LogSoftmax(dim=1)
            )


        if group_blocks == 3:
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns, out_channels=32, conv_blocks = conv_blocks),
                ConvBlock(in_channels=32, out_channels=48,conv_blocks = conv_blocks),
                ConvBlock(in_channels=48, out_channels=64,conv_blocks = conv_blocks)
            )

            self.classifier = nn.Sequential(
                #             nn.Dropout(0.2),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                #             nn.Dropout(0.2),
                nn.Linear(32, 3),
                nn.LogSoftmax(dim=1)
            )

    @property
    def group_blocks(self):
        return self._group_blocks
    
    @group_blocks.setter
    def group_blocks(self, value):
        if value < 0 or value > 3:
            print("Erro: group_blocks inserido maior do que 3. O valor padrão de 1 será usado.")
            self._group_blocks = 1
        else:
            self._group_blocks = value


    def forward(self, x):
        out = self.conv(x)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        out = nn.functional.softmax(out, dim=1)
        return out


if __name__ == "__main__":
    from helper_functions import  create_writer , save_model
    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix
    import random

    
    input_size = [250,500,1000]

    for i in range(len(input_size)):

        data = DATA_1M(seconds=40,columns=input_size[i]) ; data_fourier = data(Fourier=True, Normalizing= True)
        
        Torch = NeuralNetCNN(columns= input_size[i] ,conv_blocks =6 ,groupblocks=3)

        data.Spliting(data= data_fourier, random_state= random.randint(10,100000), test_size = 0.275, shuffle = True, inplace= False)

        train_dataloader , test_dataloader = data.DataLoaders(batch_size=512, inplace=True)

        experiment_name = f"Fourier_{i}_{data_fourier.shape[1]-1}_inputsize_{len(list(Torch.Cnn.children()))}_Layers_{Torch.count_blocks(Torch.Cnn)}_Blocks"
        
        model_name = f"CNN_19TEP_Fourier_10_Layers_{i}"

        print(f"Experiment Fourier:  {data_fourier.shape[1]-1} input size ,  {len(list(Torch.Cnn.children()))} Layers ,{Torch.count_blocks(Torch.Cnn)} Blocks")

        
        Torch.training_loop(model= Torch.Cnn,
            data_loader_train= train_dataloader,
            data_loader_test= test_dataloader,
            optimizer= Torch.optimizer,
            loss_fn= Torch.loss_fn,
            device= Torch.device,
            accuracy_fn= accuracy_fn,
            writer= create_writer(experiment_name=experiment_name,
                                model_name= model_name
                                ),
            epochs= 30)
        ## Salvando o modelo
        save_filepath = f"Model _{model_name}.pth"
        save_model(model=Torch.Cnn,
                target_dir="models",
                model_name=save_filepath)
        class_names =["CLEAR", "WIFI", "LTE"]
        y_pred_tensor = Torch.Making_Predictions(model=Torch.Cnn,data_loader=test_dataloader)
        
        # Plot the confusion matrix
        print(classification_report(data.y_test,y_pred_tensor,
                                    target_names= class_names))
        
        Torch.print_confusion_matrix(y_pred= y_pred_tensor, y_true= data.y_test, num_classes= 3)

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
