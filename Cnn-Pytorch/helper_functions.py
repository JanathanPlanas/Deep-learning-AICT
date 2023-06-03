"""
A series of helper functions used throughout the Pytorch Model.


"""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pip
import requests
import seaborn as sns
import torch
import torchmetrics
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV
# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

writer = SummaryWriter()
device = "cuda" if torch.cuda.is_available() else "cpu"


"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Salva um modelo PyTorch em um diretório de destino.

     Argumentos:
     model: Um modelo PyTorch de destino para salvar.
     target_dir: Um diretório para salvar o modelo.
     model_name: Um nome de arquivo para o modelo salvo. Deveria incluir
       ".pth" ou ".pt" como a extensão do arquivo.

     Exemplo de uso:
     save_model(model=model_0,
                target_dir="modelos",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Treina um modelo PyTorch para uma única época.

     Transforma um modelo PyTorch de destino no modo de treinamento e, em seguida,
     percorre todas as etapas de treinamento necessárias (avançar
     passar, cálculo de perda, etapa do otimizador).

     Argumentos:
     model: um modelo PyTorch a ser treinado.
     dataloader: Uma instância do DataLoader para o modelo a ser treinado.
     loss_fn: uma função de perda do PyTorch para minimizar.
     otimizador: Um otimizador PyTorch para ajudar a minimizar a função de perda.
     dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").

     Retorna:
     Uma tupla de métricas de perda de treinamento e precisão de treinamento.
     No formulário (train_loss, train_accuracy). Por exemplo:

     (0,1112, 0,8743)
     """
    # Put model in train mode
    model.to(device)
    model.train().double()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (inputs, target) in enumerate(dataloader):
        # movendo os dados para o dispositivo de processamento
        inputs, target = inputs.to(device), target.to(device)
        inputs = inputs.unsqueeze(2)

        # fazendo as previsões
        target_pred = model(inputs.double())

        # calculando a perda
        loss = loss_fn(target_pred, target.long())

        train_loss += loss.data.item()

        # retropropagando os gradientes e atualizando os pesos
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(target_pred, dim=1), dim=1)
        train_acc += (y_pred_class == target).sum().data.item() / \
            len(target_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Testa um modelo PyTorch para uma única época.

     Transforma um modelo PyTorch de destino no modo "eval" e, em seguida, executa
     uma passagem direta em um conjunto de dados de teste.

     Argumentos:
     model: Um modelo PyTorch a ser testado.
     dataloader: Uma instância do DataLoader para o modelo a ser testado.
     loss_fn: uma função de perda do PyTorch para calcular a perda nos dados de teste.
     dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").

     Retorna:
     Uma tupla de perda de teste e métricas de precisão de teste.
     No formulário (test_loss, test_accuracy). Por exemplo:

     (0,0223, 0,8985)
     """
    # Coloca o modelo no modo eval
    model.eval().double()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (data, target) in enumerate(dataloader):
            # Send data to target device
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(2)

            # 1. Forward pass
            test_pred_logits = model(data)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, target.long())
            test_loss += loss.data.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels ==
                         target).sum().data.item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def Making_Predictions(data_loader: torch.utils.data.DataLoader,
                           model: torch.nn.Module,
                           device : torch.device):
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
                X, y = X.to(device), y.to(device)

                X = X.unsqueeze(2)
                # Do the forward pass
                y_logit = model(X)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())
            # Concatenate list of predictions into a tensor
            y_pred_tensor = torch.cat(y_preds)

        return y_pred_tensor


def train(model: torch.nn.Module,
          data_loader_train: torch.utils.data.DataLoader,
          data_loader_test: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
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
    # Cria um dicionário de resultados vazio
    results = {
               "test_loss": [],
               "test_acc": []
               }

    # Make sure model on target device
    model.to(device).double()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=data_loader_train,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=data_loader_test,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"Test Loss": test_loss},
                               global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"Test Accuracy": test_acc},
                               global_step=epoch)

            # Track the PyTorch model architecture
            writer.add_graph(model=model,
                             # Pass in an example input
                             input_to_model=torch.randn(8, 2000, 1).to(device).double())
            

            # Close the writer
            writer.close()
        else:
            pass

    # Return the filled results at the end of the epochs
    return results




def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def normalize(data):

    scaler = StandardScaler()
    scaler.fit(data)

    return scaler.transform(data)


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(
        np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(
        str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path


mm = MinMaxScaler(feature_range=(0, 1))


class Visualization:
    def __init__(self):
        pass

    def plot_pie(self, y_test):
        counts = y_test.value_counts(normalize=True)
        counts.plot.pie(autopct="%0.2f%%")
        plt.show()


class RandomizedSearchCVWrapper:

    def __init__(self, model, param_distributions, X_train, y_train,
                 n_iter, cv, random_state, n_jobs,
                 scoring, verbose, return_train_score) -> None:

        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose
        self.return_train_score = return_train_score

        self.randomsearch = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv,
                                               random_state=random_state, n_jobs=n_jobs, scoring=scoring,
                                               verbose=verbose, return_train_score=return_train_score)

        self.random_fit = self.randomsearch.fit(
            mm.fit_transform(X_train), y_train)
        self.results = pd.DataFrame(self.random_fit.cv_results_).sort_values(
            "mean_test_score", ascending=False)
        self.best_params = self.random_fit.best_params_

    def plot_mean_performance(self):

        plt.figure(figsize=(16, 4))
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["mean_test_score"]), label="Testing Score", linestyle='dotted')
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["mean_train_score"]), label="Training Score", linestyle='dotted')
        plt.xlabel("Number of N neighbors ")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.title("Performance vs Number of K")

    def plot_std_performance(self):

        plt.figure(figsize=(16, 4))
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["std_test_score"]), label="Testing Error", linestyle='dotted')
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["std_train_score"]), label="Training Error", linestyle='dotted')
        plt.xlabel("Number of N neighbors ")
        plt.ylabel("Root from the Mean Absolute Error")
        plt.legend()
        plt.title("Performance vs Number of K")
