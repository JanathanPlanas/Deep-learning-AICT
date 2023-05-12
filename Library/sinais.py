
import numpy as np
import pandas as pd


class signal:

    def __init__(self, data, slice, sinal, id) -> None:
        self.path = data  # dataset numpy
        self.data_sliced = slice  # size limit the dataset by number of rows by user
        self.type = self.path.dtype  # dtype
        self.sinal = sinal  # Naming the signal captured
        self.id = id

    def info(self) -> None:

        self.new_data = self.path[:self.data_sliced]  # new data after slicing
        self.length = (self.new_data).shape  # shape from the dataset

        # Applying Discrete Transform Fourier
        self.fourier = np.fft.fftn(self.new_data)
        self.abs = np.abs(self.fourier)  # Applying the module from FFT
        # slices the data only the real numbers
        self.real = np.real(self.new_data)
        # slices the data only the imaginary numbers
        self.imag = np.imag(self.new_data)

        print((pd.DataFrame(self.new_data)).info())

    @property
    def id(self):  # Picking a number wich means the captured signal
        return self._id

    @id.setter
    def id(self, value):
        if isinstance(value, str):
            self._id = value
        else:
            raise Exception("Input integer type ID")

    def to_pandas(self) -> None:

        self.database = pd.DataFrame(self.real, self.imag,
                                     dtype="float64[pyarrow]").reset_index().rename(columns={'index': f"{self.sinal}_Real",
                                                                                             0: f"{self.sinal}_Imag"})
        self.database['ID'] = self.id


class concat_3(signal):

    def __init__(self, database1, database2, database3, sinal1, sinal2, sinal3) -> None:

        self.database1 = database1
        self.database2 = database2
        self.database3 = database3
        self.sinal1 = sinal1
        self.sinal2 = sinal2
        self.sinal3 = sinal3

    def merge(self) -> None:

        print("Concatenação efetuada")
        self.rename1 = self.database1.rename(
            columns={f"{self.sinal1}_Real": "Real", f"{self.sinal1}_Imag": "Imag"})

        self.rename2 = self.database2.rename(
            columns={f"{self.sinal2}_Real": "Real", f"{self.sinal2}_Imag": "Imag"})

        self.rename3 = self.database3.rename(
            columns={f"{self.sinal3}_Real": "Real", f"{self.sinal3}_Imag": "Imag"})

        self.concat = pd.concat([self.rename1, self.rename2, self.rename3], keys=[
                                f"{self.sinal1}", f"{self.sinal2}", f"{self.sinal3}"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                                                                     'level_1': 'index'}).drop(columns=['index'])


class concat_2(signal):

    def __init__(self, database1, database2, sinal1, sinal2) -> None:

        self.database1 = database1
        self.database2 = database2
        self.sinal1 = sinal1
        self.sinal2 = sinal2

    def merge(self) -> None:

        print("Concatenação efetuada")
        self.rename1 = self.database1.rename(
            columns={f"{self.sinal1}_Real": "Real", f"{self.sinal1}_Imag": "Imag"})

        self.rename2 = self.database2.rename(
            columns={f"{self.sinal2}_Real": "Real", f"{self.sinal2}_Imag": "Imag"})

        self.concat = pd.concat([self.rename1, self.rename2], keys=[
                                f"{self.sinal1}", f"{self.sinal2}"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                                                   'level_1': 'index'}).drop(columns=['index'])
