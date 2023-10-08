
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from helper_functions import normalize

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def normalize(data):

    scaler = StandardScaler()
    scaler.fit(data)

    return scaler.transform(data)


class Loaders():

    def __init__(self, batch_size: int, normalization: bool, file_path='future_clean/future-1/') -> None:

        self.normalize = normalization

        self.file_path = file_path
        data_paths = ['train.parquet', 'val.parquet', 'test.parquet']
        data_dict = {}

        for data_type, data_path in zip(['train', 'val', 'test'], data_paths):
            df = pd.read_parquet(f'{self.file_path }{data_path}')
            data_dict[data_type] = df.to_numpy()

        self.train = (data_dict['train'])
        self.val = (data_dict['val'])
        self.test = (data_dict['test'])

        if normalization == True:
            X_train = normalize(self.train[:, :-1])
            y_train = self.train[:, -1]
            X_test = normalize(self.test[:, :-1])
            y_test = self.test[:, -1]
            X_val = normalize(self.val[:, :-1])
            y_val = self.val[:, -1]

        if normalization == False:
            X_train = (self.train[:, :-1])
            y_train = self.train[:, -1]
            X_test = (self.test[:, :-1])
            y_test = self.test[:, -1]
            X_val = (self.val[:, :-1])
            y_val = self.val[:, -1]

        numpy_list = [X_train, y_train, X_val, y_val, X_test, y_test]
        tensor_list = []

        for npy in numpy_list:
            tensor = torch.Tensor(npy).to(device, dtype=torch.float64)
            tensor_list.append(tensor)

        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = tensor_list

        datasets = [train_dataset, val_dataset, test_dataset] = [
            TensorDataset(X_train_tensor, y_train_tensor),
            TensorDataset(X_val_tensor, y_val_tensor),
            TensorDataset(X_test_tensor, y_test_tensor)
        ]

        loaders = [train_loader, val_loader, test_loader] = [
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            for dataset, shuffle in zip(datasets, [True, False, True])
        ]

        self.train_loader, self.val_loader, self.test_loader = loaders

    def __setitem__(self, value):

        self.file_path = value  # setar os valor de linhas quando o usuario quiser

    # objeto Call usado para escolher se o dataset terá ou não transformada de Fourier

    def get_path(self):
        """
        Retorna o atributo 'clear'.

        Returns:
            numpy.ndarray: O array 'clear'.
        """
        return self.file_path

    def __call__(self, Fourier=False, Normalizing=False):
        """
        Retorna o dataframe após aplicar a Transformada de Fourier e/ou a normalização nos dados.

        Args:
            Fourier (bool, optional): Indica se a Transformada de Fourier deve ser aplicada. O padrão é False.
            Normalizing (bool, optional): Indica se a normalização deve ser aplicada. O padrão é False.

        Returns:
            numpy.ndarray: O dataframe resultante.
        """

        if Fourier == True and Normalizing == True:

            """
            Retorna o dateframe apos a Tranformada de Fourier.
            """
            empyt_list = []
            # Aplicando a transformada de Fourier
            empyt_list = list(map(lambda x: [np.fft.fftn(x)], self.sinais))

            # SEPARANDO REAL E IMAG
            clear, lte, wifi = map(lambda x:  np.concatenate((np.real(x).reshape(-1, 1), np.imag(
                x).reshape(-1, 1)), axis=1), empyt_list[:3])

            # APLICANDO NORMALIZAÇÃO
            clear = normalize(data=clear)
            lte = normalize(data=lte)
            wifi = normalize(data=wifi)

            samples_signal = [clear, lte, wifi]

            # RESHAPE DOS DADOS

            signal_list = list(map(lambda x: np.hstack(
                x).reshape(-1, self.values_reshaped), samples_signal))
            # Reshaping data            # Slicing data

        elif Fourier == False and Normalizing == True:

         # Aplicando a transformada de Fourier

            clear, lte, wifi = map(lambda x:  np.concatenate((np.real(x).reshape(-1, 1), np.imag(
                x).reshape(-1, 1)), axis=1), self.sinais[:3])  # Spliting the real and imaginary numbers

            clear = normalize(data=clear)
            lte = normalize(data=lte)
            wifi = normalize(data=wifi)

            samples_signal = [clear, lte, wifi]

            signal_list = list(map(lambda x: np.hstack(
                x).reshape(-1, self.values_reshaped), samples_signal))

        elif Fourier == True and Normalizing == False:

            empyt_list = []
            # Aplicando a transformada de Fourier
            empyt_list = list(map(lambda x: [np.fft.fftn(x)], self.sinais))

            # SEPARANDO REAL E IMAG
            clear, lte, wifi = map(lambda x:  np.concatenate((np.real(x).reshape(-1, 1), np.imag(
                x).reshape(-1, 1)), axis=1), empyt_list[:3])  # Spliting the real and imaginary numbers

            samples_signal = [clear, lte, wifi]

            signal_list = list(map(lambda x: np.hstack(
                x).reshape(-1, self.values_reshaped), samples_signal))

# Slicing data

        # Creating labels  0 - CLEAR , 1 - WIFI - 2 LTE
        result = []
        for i, signal in enumerate(signal_list):
            if i == 0:
                # Label 0 = Clear
                arr = np.zeros((signal.shape[0], signal.shape[1] + 1))
                arr[:, :-1] = signal
            elif i == 1:
                # Label 1 = WIFI
                arr = np.ones((signal.shape[0], signal.shape[1] + 1))
                arr[:, :-1] = signal
            else:
                col = np.full((signal.shape[0], 1), 2)  # Label 2 = LTE
                arr = np.hstack((signal, col))

            result.append(arr)

        self.full_dataset = np.concatenate(result, axis=0)

        print(
            f"tamanho da memória ocupada :{self.full_dataset.nbytes/(1024**2):.2f} MB")

        return np.concatenate(result, axis=0)

    # def Spliting(self, data: np.ndarray, random_state, test_size, shuffle: bool, inplace: bool):
    #     """
    #     Realiza a divisão dos dados em conjunto de treinamento e teste.

    #     Args:
    #         data (np.ndarray): O dataframe a ser dividido.
    #         random_state: O valor de semente aleatória para reprodução dos resultados.
    #         test_size: A proporção do conjunto de teste em relação ao conjunto total.
    #         shuffle (bool): Indica se os dados devem ser embaralhados antes da divisão.
    #         inplace (bool): Indica se os conjuntos divididos devem substituir os atributos da classe ou serem retornados.

    #     Returns:
    #         tuple: Uma tupla contendo os conjuntos de treinamento e teste, respectivamente.

    #     """

    #     X = data[:, :-1]
    #     y = data[:, -1]

    #     X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                         test_size=test_size,
    #                                                         random_state=random_state, shuffle=shuffle)

    #     # Convertendos os arrays para tensor e passando para device setadi
    #     self.X_train = torch.tensor(X_train).to(device)
    #     self.X_test = torch.tensor(X_test).to(device)
    #     self.y_train = torch.tensor(y_train).to(device)
    #     self.y_test = torch.tensor(y_test).to(device)

    #     print("X_train shape:", X_train.shape, X_train.dtype)
    #     print("X_Test shape:", X_test.shape, X_test.dtype)
    #     print("y_train shape:", y_train.shape, y_test.dtype)
    #     print("y_test shape:", y_test.shape, y_train.dtype)
    #     print("\n--------")

    #     print("X_train device:", self.X_train.device)
    #     print("X_Test device:", self.X_test.device)
    #     print("y_train device:", self.y_train.device)
    #     print("y_test device:", self.y_test.device)

    #     arr_non_negative = y.astype('int64') - np.min(y.astype('int64'))
    # # Calcular a contagem de cada valor
    #     counts = np.bincount(arr_non_negative)

    #     # Imprimir a contagem de cada valor
    #     for value, count in enumerate(counts):
    #         print(
    #             f"Valor {value}: {count} ocorrência(s)- {round(count/np.sum(counts),2)}%")

    #     print("Dataset : ", self.full_dataset.shape)

    #     if inplace == True:
    #         return X_train, X_test, y_train, y_test
    #     else:
    #         pass

        # convertendo numpy arrays em tensores do PyTorch
