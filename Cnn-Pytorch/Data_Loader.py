
from typing import Any

import numpy as np
import pandas as pd
import torch
from helper_functions import normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

LABEL = {'CLEAR': 0, 'WIFI': 1, "LTE": 2}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DATA_1M():

    """
    Classe para manipulação de dados do dataset 1M.

    Args:
        seconds (int): Número de segundos a serem tomados/repartidos do dataset.
        columns (int): Quantidade de colunas a serem feitas no reshape.
        jump_time (int): Valor em segundos que a função vai 'pular' o dataset.
        n_jumps (int): Quantidade de vezes que será 'pulado' o dataset.

    Attributes:
        sec (int): Número de segundos que será tomado/repartido do dataset.
        jump (int): Valor em segundos que a função vai 'pular' o dataset.
        values_reshaped (int): Quantidade de colunas a serem feitas no reshape.
        n_jumps (int): Quantidade de vezes que será 'pulado' o dataset.
        signal (list): Lista contendo os dados dos sinais carregados a partir de arquivos.
        clear (ndarray): Dados do sinal 'Clear'.
        wifi (ndarray): Dados do sinal 'WIFI_1M'.
        lte (ndarray): Dados do sinal 'LTE_1M'.
        row (int): Número de linhas calculado a partir dos parâmetros de segundos e colunas.
        jump_time_rows (int): Número de linhas calculado a partir do parâmetro de 'pulos'.
    """

    def __init__(self, seconds, columns) -> None:

        self.sec = seconds  # numero de segundos que sera tomado/repartido do dataset

        self.values_reshaped = columns  # quantidade de colunas a ser feito reshapee

        data = ['Clear.npy', 'WIFI_1M.npy', 'LTE_1M.npy']
        self.signal = list(map(lambda x: (np.load(
            f"C:/Users/janathan.pena/Documents/GitHub/{x}")), data))  # Data load

        

        # transformando os parametros de segundos para número de linhas
        self.row = int(
            np.round((self.sec*(self.signal[0].shape[0]) / (90)) / columns)*columns)

        # transformando os parametros de 'pulos' para número de linhas
        
        
        self.clear_ = self.signal[0][: self.row]
        self.wifi_ = self.signal[1][: self.row]
        self.lte_ = self.signal[2][: self.row]
        
        self.sinais = [self.clear_, self.wifi_ ,self.lte_]

    def __str__(self) -> str:

        """
        Retorna uma representação em string dos parâmetros da classe.

        Returns:
            str: Representação em string dos parâmetros da classe.
        """
        return f"Número de linhas {self.row}, equivalente a {self.sec} segundos \n Pulando em {self.jump} segundos em {self.n_jumps} vezes"

    def __setitem__(self, value):

        self.row = value  # setar os valor de linhas quando o usuario quiser

    # objeto Call usado para escolher se o dataset terá ou não transformada de Fourier

    def get_clear(self):
        """
        Retorna o atributo 'clear'.

        Returns:
            numpy.ndarray: O array 'clear'.
        """
        return self.clear

    def get_wifi(self):
        """
        Retorna o atributo 'wifi'.

        Returns:
            numpy.ndarray: O array 'wifi'.
        """
        return self.wifi

    def get_lte(self):
        """
        Retorna o atributo 'lte'.

        Returns:
            numpy.ndarray: O array 'lte'.
        """
        return self.lte

    

    def __getattribute__(self, item):

        """
        Sobrescreve o método __getattribute__ para imprimir informações sobre os atributos 'clear', 'wifi' e 'lte'
        quando eles forem acessados.

        Args:
            item (str): Nome do atributo a ser acessado.

        Returns:
            O valor do atributo, se for diferente de 'clear', 'wifi' ou 'lte'.
        """
        if item == 'clear':
            print(
                """shape array {},\n
                Clear array \n {} \n  
                memory usage {} MB""".format(
                    object.__getattribute__(self, item).shape,
                    object.__getattribute__(self, item),
                    "%.2f" % (object.__getattribute__(
                        self, item).nbytes / (1024 ** 2))
                )
            )

        elif item == 'lte':
            print(
                """shape array {},\n
                LTE array \n{} \n  
                memory usage {} MB""".format(
                    object.__getattribute__(self, item).shape,
                    object.__getattribute__(self, item),
                    "%.2f" % (object.__getattribute__(
                        self, item).nbytes / (1024 ** 2))
                )
            )

        elif item == 'wifi':
            print(
                """shape array {},\n
               WIFI array \n {} \n  
                memory usage {} MB""".format(
                    object.__getattribute__(self, item).shape,
                    object.__getattribute__(self, item),
                    "%.2f" % (object.__getattribute__(
                        self, item).nbytes / (1024 ** 2))
                )
            )
        else:
            return object.__getattribute__(self, item)

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
            empyt_list = list(map(lambda x: [np.fft.fftn(x)], self.sinais))  # Aplicando a transformada de Fourier

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
            empyt_list = list(map(lambda x: [np.fft.fftn(x)], self.sinais))  # Aplicando a transformada de Fourier

            # SEPARANDO REAL E IMAG
            clear, lte, wifi = map(lambda x:  np.concatenate((np.real(x).reshape(-1, 1), np.imag(
                x).reshape(-1, 1)), axis=1), empyt_list[:3]) # Spliting the real and imaginary numbers

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

    def Spliting(self, data: np.ndarray, random_state, test_size, shuffle: bool, inplace: bool):


        """
        Realiza a divisão dos dados em conjunto de treinamento e teste.

        Args:
            data (np.ndarray): O dataframe a ser dividido.
            random_state: O valor de semente aleatória para reprodução dos resultados.
            test_size: A proporção do conjunto de teste em relação ao conjunto total.
            shuffle (bool): Indica se os dados devem ser embaralhados antes da divisão.
            inplace (bool): Indica se os conjuntos divididos devem substituir os atributos da classe ou serem retornados.

        Returns:
            tuple: Uma tupla contendo os conjuntos de treinamento e teste, respectivamente.

        """

        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state, shuffle=shuffle)

        # Convertendos os arrays para tensor e passando para device setadi
        self.X_train = torch.tensor(X_train).to(device)
        self.X_test = torch.tensor(X_test).to(device)
        self.y_train = torch.tensor(y_train).to(device)
        self.y_test = torch.tensor(y_test).to(device)

        print("X_train shape:", X_train.shape, X_train.dtype)
        print("X_Test shape:", X_test.shape, X_test.dtype)
        print("y_train shape:", y_train.shape, y_test.dtype)
        print("y_test shape:", y_test.shape, y_train.dtype)
        print("\n--------")

        print("X_train device:", self.X_train.device)
        print("X_Test device:", self.X_test.device)
        print("y_train device:", self.y_train.device)
        print("y_test device:", self.y_test.device)

        arr_non_negative = y.astype('int64') - np.min(y.astype('int64'))
    # Calcular a contagem de cada valor
        counts = np.bincount(arr_non_negative)

        # Imprimir a contagem de cada valor
        for value, count in enumerate(counts):
            print(
                f"Valor {value}: {count} ocorrência(s)- {round(count/np.sum(counts),2)}%")

        print("Dataset : ", self.full_dataset.shape)

        if inplace == True:
            return X_train, X_test, y_train, y_test
        else:
            pass

        # convertendo numpy arrays em tensores do PyTorch
    def DataLoaders(self, batch_size, inplace: bool):

        """
        Cria os dataloaders para o conjunto de treinamento e teste.

        Args:
            batch_size: O tamanho de lote para cada iteração do dataloader.
            inplace (bool): Indica se os dataloaders criados devem substituir os atributos da classe ou serem retornados.

        Returns:
            tuple: Uma tupla contendo os dataloaders de treinamento e teste, respectivamente.

        """ 

        # criando datasets
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

        # criando dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)

        X, y = next(iter(self.train_dataloader))
        # Visualizar o shape do dataloader target e treino
        print(f"X {X.shape} y {y.shape}")
        print("----------------\n")

        print(f"Dataloaders: {self.train_dataloader, self.test_dataloader}")
        # Visualizar o size de cada batch, como foi feito a repartição dos dados
        print(
            f"Length of train dataloader: {len(self.train_dataloader)} batches of {batch_size}")
        print(
            f"Length of test dataloader: {len(self.test_dataloader)} batches of {batch_size}")
        if inplace == True:
            return self.train_dataloader, self.test_dataloader
        else:
            pass

