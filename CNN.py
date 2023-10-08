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
import torch.nn.functional as F
import torch.optim as optim
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm

from helper_functions import accuracy_fnn


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

        if conv_blocks >= 2:  # 2 layer com 4 subcamadas

            self.conv2 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        if conv_blocks >= 3:  # 3 layer com 4 subcamadas

            self.conv3 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Tanh(),
                nn.MaxPool1d(kernel_size=2)
            )

        if conv_blocks >= 4:  # 4 layer com 4 subcamadas

            self.conv4 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Tanh(),
                nn.MaxPool1d(kernel_size=2)
            )

        if conv_blocks >= 5:  # 5 layer com 4 subcamadas

            self.conv5 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Tanh(),
                nn.MaxPool1d(kernel_size=2)
            )
        if conv_blocks == 6:  # 5 layer com 4 subcamadas

            self.conv6 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=2, padding=1, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Tanh(),
                nn.MaxPool1d(kernel_size=2)
            )

    def forward(self, x):

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

    def __init__(self, in_channels_columns, conv_blocks, group_blocks):
        super().__init__(in_channels=in_channels_columns,
                         out_channels=32, conv_blocks=conv_blocks)

        if group_blocks == 1:  # 1 layer com 4 subcamadas
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns,
                          out_channels=32, conv_blocks=conv_blocks),
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

        if group_blocks == 2:  # 2 layer com 4 subcamadas
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns,
                          out_channels=32, conv_blocks=conv_blocks),
                ConvBlock(in_channels=32, out_channels=48,
                          conv_blocks=conv_blocks),
            )

            self.classifier = nn.Sequential(
                #             nn.Dropout(0.2),
                nn.BatchNorm1d(48),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                #             nn.Dropout(0.2),
                nn.Linear(32, 2),
                nn.LogSoftmax(dim=1)
            )

        if group_blocks == 3:
            self.conv = nn.Sequential(
                ConvBlock(in_channels=in_channels_columns,
                          out_channels=32, conv_blocks=conv_blocks),
                ConvBlock(in_channels=32, out_channels=48,
                          conv_blocks=conv_blocks),
                ConvBlock(in_channels=48, out_channels=64,
                          conv_blocks=conv_blocks)
            )

            self.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                # nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                #             nn.Dropout(0.2),
                nn.Linear(32, 3),
                nn.LogSoftmax(dim=1)
            )


class CNNModel(nn.Module):

    def __init__(self, in_channels):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=32,
                      kernel_size=2,
                      padding=1, stride=1),

            nn.BatchNorm1d(32),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      padding=1, stride=1),

            nn.MaxPool1d(kernel_size=2, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=2,
                      padding=1, stride=1),

            nn.BatchNorm1d(32),

            nn.Conv1d(in_channels=32,
                      out_channels=16,
                      kernel_size=2,
                      padding=1, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=3))

        self.classifier = nn.Sequential(

            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)


        )

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        out = nn.functional.softmax(out, dim=1)

        return out


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
