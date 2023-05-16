
import numpy as np

LABEL = {'CLEAR':0,'WIFI':1, "LTE":2}


class DATA_1M(): 

    def __init__(self,seconds,columns, jump_time , n_jumps) -> None:

        self.sec = seconds
        self.jump = jump_time
        self.values_reshaped = columns
        self.n_jumps = n_jumps

    def get_data(self):
                
        data= ['CLEAR_40SEC.npy','LTE_1M_40SEC.npy','WIFI1M_40SEC.npy']
        signal = []
        i = 0
        for i in range(len(data)):
            signal.append(np.load(f"Data_\{data[i]}"))



        self.row = int(np.round( (self.sec*(signal[0].shape[0]) / (18*60)) /20000 )*20000 )

        jump_time_rows = int(self.jump *(signal[0].shape[0]) / (18*60))

        clear = np.concatenate((signal[0].real.reshape(-1,1)
                                ,signal[0].imag.reshape(-1,1)), 
                                axis= 1)
        
        lte = np.concatenate((signal[1].real.reshape(-1,1)
                                ,signal[1].imag.reshape(-1,1)),
                                axis= 1)
        
        wifi = np.concatenate((signal[2].real.reshape(-1,1)
                                ,signal[2].imag.reshape(-1,1)),
                                axis= 1)

        
        clear_1 = [] ; lte_1 = [] ; wifi_1 = []
        
        
        for i in range(self.n_jumps):
                
                clear_1.append(clear[jump_time_rows:jump_time_rows+int(self.row/self.n_jumps)])

                lte_1.append(lte[jump_time_rows:jump_time_rows+int(self.row/self.n_jumps)])

                wifi_1.append(wifi[jump_time_rows:jump_time_rows+int(self.row/self.n_jumps)])




        new_clear = np.hstack(clear_1).reshape(-1,self.values_reshaped)
        new_lte = np.hstack(lte_1).reshape(-1,self.values_reshaped)
        new_wifi = np.hstack(wifi_1).reshape(-1,self.values_reshaped)
        


        # Adicionando uma coluna com valores zero
        clear = np.zeros((new_clear.shape[0], new_clear.shape[1] + 1)) ; clear[:, :-1] = new_clear  
        # Copiando os valores do array original para o novo array
        

        lte = np.ones((new_lte.shape[0], new_lte.shape[1] + 1)) ; lte[:, :-1] = new_lte
        

        col = np.full((new_wifi.shape[0], 1), 2) ;   wifi = np.hstack((new_wifi, col))

# Concatenar o array col com o arr
        full_dataset = np.concatenate((clear, lte,wifi), axis =0 )

        return full_dataset
    


if __name__== "__main__":
    
    import pandas as pd
    test= DATA_1M(seconds= 42 ,columns =2000, jump_time = 240 , n_jumps = 2)
    print(pd.DataFrame(test.get_data()[:,-1]))


        