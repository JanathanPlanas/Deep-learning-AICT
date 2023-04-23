import torch
import torch.nn as nn
import torch.optim as optim

# definir uma classe para o modelo
class MeuModelo(nn.Module):
    def __init__(self):
        super(MeuModelo, self).__init__()
        # definir as camadas do modelo
        self.camada1 = nn.Linear(10, 20)
        self.camada2 = nn.Linear(20, 1)

    def forward(self, x):
        # definir como os dados fluem pelo modelo
        x = torch.relu(self.camada1(x))
        x = self.camada2(x)
        return x

# criar instância do modelo
modelo = MeuModelo()

# definir critério de perda e otimizador
criterio = nn.MSELoss()
otimizador = optim.SGD(modelo.parameters(), lr=0.01)

# criar dados de entrada e saída
entrada = torch.randn(32, 10)
saida = torch.randn(32, 1)

# treinar o modelo por algumas épocas
for epoch in range(10):
    # calcular saída do modelo e perda
    saida_predita = modelo(entrada)
    perda = criterio(saida_predita, saida)

    # zerar os gradientes do otimizador, calcular gradientes e atualizar parâmetros
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    # imprimir perda a cada época
    print('Época {}: Perda = {}'.format(epoch, perda.item()))
