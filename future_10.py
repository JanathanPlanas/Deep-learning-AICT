import numpy as np
import pandas as pd


def building_data(test, column_name='unit1_pwr_60ghz_1'):
    """
    Constrói um DataFrame a partir de arquivos CSV com base em uma coluna específica
    e adiciona a coluna 'blockage_final' ao DataFrame.

    Args:
        test (pandas.DataFrame): O DataFrame de entrada contendo as colunas de dados.
        column_name (str): O nome da coluna a ser lida nos arquivos CSV.

    Returns:
        pandas.DataFrame: O DataFrame resultante com os dados concatenados e 'blockage_final' adicionada.

    Exemplo:
        >>> test_data = pd.DataFrame({'unit1_pwr_60ghz_1': ['file1.csv', 'file2.csv']})
        >>> test_data['blockage_final'] = [1, 2]
        >>> result = building_data(test_data, 'unit1_pwr_60ghz_1')
        >>> print(result.head())
           unit1_pwr_60ghz_1  blockage_final
        0                0.001               1
        1                0.002               1
        2                0.003               2
        3                0.004               2
        """
    dfs = []

    for i in range(len(test[column_name])):
        df = pd.read_csv(test[column_name][i], dtype=np.double,
                         header=None, engine='pyarrow', dtype_backend='pyarrow')
        df['blockage_1'] = test['blockage_1'][i]
        df['blockage_2'] = test['blockage_2'][i]
        df['blockage_3'] = test['blockage_3'][i]
        df['blockage_4'] = test['blockage_4'][i]
        df['blockage_5'] = test['blockage_5'][i]
        df['blockage_6'] = test['blockage_6'][i]
        df['blockage_7'] = test['blockage_7'][i]
        df['blockage_8'] = test['blockage_8'][i]
        df['blockage_9'] = test['blockage_9'][i]
        df['blockage_10'] = test['blockage_10'][i]
        dfs.append(df)

    return pd.concat(dfs, axis=0).rename(columns={0: column_name}).reset_index(drop=True)


def merge_data(test):
    """
    Mescla múltiplos DataFrames construídos usando a função 'building_data' em uma única estrutura.

    Args:
        test (pandas.DataFrame): O DataFrame contendo as colunas de dados.

    Returns:
        pandas.DataFrame: O DataFrame resultante com as colunas mescladas.

    Exemplo:
        >>> test_data = pd.DataFrame({'unit1_pwr_60ghz_1': ['file1.csv', 'file2.csv'],
        ...                           'unit1_pwr_60ghz_2': ['file3.csv', 'file4.csv']})
        >>> test_data['blockage_final'] = [1, 2]
        >>> result = merge_data(test_data)
        >>> print(result.head())
           unit1_pwr_60ghz_1  unit1_pwr_60ghz_2  blockage_final
        0                0.001                0.001               1
        1                0.002                0.002               1
        2                0.003                0.003               2
        3                0.004                0.004               2

    """
    columns = ['unit1_pwr_60ghz_1', 'unit1_pwr_60ghz_2', 'unit1_pwr_60ghz_3',
               'unit1_pwr_60ghz_4', 'unit1_pwr_60ghz_5', 'unit1_pwr_60ghz_6',
               'unit1_pwr_60ghz_7', 'unit1_pwr_60ghz_8']

    data_frames = [building_data(test, column_name) for column_name in columns]

    concat_data_frames = [pd.concat(data_frames[i:i + 2], axis=1)
                          for i in range(0, len(data_frames), 2)]

    return pd.concat(concat_data_frames, axis=1)

# Exemplo de uso:
# result = merge_data(test_data)  # Substitua 'test_data' pelos seus dados de teste

# Exemplo de uso:
# result = merge_data(test_data)  # Substitua 'test_data' pelos seus dados de teste


if __name__ == "__main__":

    test = pd.read_csv(
        r'future-10\scenario17_dev_series_train.csv')
    df_test = merge_data(test)
    df_test = merge_data(test).drop(columns=['blockage_1',
                                             'blockage_2',
                                             'blockage_3',
                                             'blockage_4',
                                             'blockage_5',
                                             'blockage_6',
                                             'blockage_7',
                                             'blockage_8',
                                             'blockage_9',
                                             'blockage_10'])

    blockage_final_column = building_data(test, 'unit1_pwr_60ghz_1')[['blockage_1',
                                                                      'blockage_2',
                                                                      'blockage_3',
                                                                      'blockage_4',
                                                                      'blockage_5',
                                                                      'blockage_6',
                                                                      'blockage_7',
                                                                      'blockage_8',
                                                                      'blockage_9',
                                                                      'blockage_10']]
    test_final = pd.concat([df_test, blockage_final_column], axis=1)
    test_final.to_parquet(r'future_clean\future-10\train.parquet')
