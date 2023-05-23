
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

LABEL = {'CLEAR': 0, 'WIFI': 1, "LTE": 2}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DATA_1M():

    def __init__(self, seconds, columns, jump_time, n_jumps) -> None:

        self.sec = seconds  # numero de segundos que sera tomado/repartido do dataset
        self.jump = jump_time  # valor em segundos que a função vai 'pular' o dataset
        self.values_reshaped = columns  # quantidade de colunas a ser feito reshapee
        self.n_jumps = n_jumps  # quantidos de vezes que será 'pulado' o dataset

        data = ['Clear.npy', 'WIFI_1M.npy', 'LTE_1M.npy']
        self.signal = list(map(lambda x: (np.load(
            f"C:/Users/janat/OneDrive/Documentos/GitHub/Data_/{x}")), data))  # Data load

        # transformando os parametros de segundos para número de linhas
        self.row = int(
            np.round((self.sec*(self.signal[0].shape[0]) / (60)) / columns)*columns)

        # transformando os parametros de 'pulos' para número de linhas
        self.jump_time_rows = int(self.jump * (self.signal[0].shape[0]) / (60))

    def __str__(self) -> str:
        return f"Número de linhas {self.row}, equivalente a {self.sec} segundos \n Pulando em {self.jump} segundos em {self.n_jumps} vezes"

    def __setitem__(self, key, value):

        self.row = value  # setar os valor de linhas quando o usuario quiser

    # objeto Call usado para escolher se o dataset terá ou não transformada de Fourier
    def __call__(self, Fourier=False):

        if Fourier == True:

            """
            Retorna o dateframe apos a Tranformada de Fourier.
            """
            empyt_list = []
            empyt_list = list(map(lambda x: [np.fft.fftn(x[self.jump_time_rows:self.jump_time_rows+int(self.row/self.n_jumps)]) for i in range(self.n_jumps)],

                                  self.signal))  # Aplicando a transformada de Fourier
            # Slicing data
        else:
            empyt_list = []           # Não aplicando a Transformada de Fourier

            empyt_list = list(map(lambda x: [(x[self.jump_time_rows:self.jump_time_rows+int(self.row/self.n_jumps)]) for i in range(self.n_jumps)],

                                  self.signal))  # Slicing data

        clear, lte, wifi = map(lambda x: np.concatenate((np.real(x).reshape(-1, 1), np.imag(
            x).reshape(-1, 1)), axis=1), empyt_list[:3])  # Spliting the real and imaginary numbers

        samples_signal = [clear, lte, wifi]

        signal_list = list(map(lambda x: np.hstack(
            x).reshape(-1, self.values_reshaped), samples_signal))  # Reshaping data

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
            f"tamanho da memória ocupada :{self.full_dataset.nbytes/1024**2:.f} MB ")

        return np.concatenate(result, axis=0)

    def Spliting(self, data: np.ndarray, random_state, test_size, shuffle: bool, inplace: bool):

        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state, shuffle=shuffle)

        # Convertendos os arrays para tensor e passando para device setadi
        self.X_train = (torch.tensor(X_train)).to(device)
        self.X_test = torch.tensor(X_test).to(device)
        self.y_train = torch.tensor(y_train).to(device)
        self.y_test = torch.tensor(y_test).to(device)

        print("X_train shape:", X_train.shape, X_train.dtype)
        print("X_Test shape:", X_test.shape, X_test.dtype)
        print("y_train shape:", y_train.shape, y_test.dtype)
        print("y_test shape:", y_test.shape, y_train.dtype)
        print("\n--------")

        print("X_train device:", X_train.device)
        print("X_Test device:", X_test.device)
        print("y_train device:", y_train.device)
        print("y_test device:", y_test.device)

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
        print(X.shape, y.shape)
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


# if __name__ == "__main__":

#      import pandas as pd

#      # Criando o objeto com os parâmetros
#      model =  DATA_1M(seconds= 20 ,columns=2000, jump_time =0 , n_jumps=1)
#      print(pd.DataFrame(model.loading_data()))
