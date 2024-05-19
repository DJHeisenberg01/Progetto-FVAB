import numpy as np
import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy.spatial.distance import euclidean, cityblock, seuclidean, canberra, mahalanobis, chebyshev, braycurtis

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

class ImageDataSet(Dataset):
    def __init__(self, img_dir, codify_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.codify_dir = codify_dir
        
        # Importazione delle immagini
        self.img_tensor = self._createImgTensor()
        
        # Importazione delle labels
        # Importa il csv della codifica in un dataframe pandas
        df = pd.read_csv(self.codify_dir, header=None)
        
        # e lo converte in un tensore
        self.img_labels = torch.Tensor(df.values)        
        
        # Eventuali trasformazioni da applicare ai dati
        self.transform = transform
        self.target_transform = target_transform
        
    
    def _createImgTensor(self):
        list_images = list()
        for file_name in os.listdir(self.img_dir):
            if not file_name.startswith('.'):
                # Carica l'immagine
                img = cv2.imread(os.path.join(self.img_dir, file_name))

                # Modifica la dimensione
                img = cv2.resize(img, (128, 128))
                
                # Converte in scala di grigio
                img = rgb2gray(img)

                # Converte in np.float32
                img_array = np.asarray(img, dtype=np.float32)
                
                # Converto OGNI immagine in un tensore 128x128 separato
                img_tensor = torch.from_numpy(img_array)
                
                # Aggiungo la dimensione relativa al canale
                img_tensor = torch.reshape(img_tensor, (1, 128, 128))
                
                # Creo una lista di tensori
                list_images.append(img_tensor)

        # Trasformo la lista di tensori in array
        tot_images = np.asarray(list_images)
        
        # E converto l'array in un unico tensore
        tot_images = torch.from_numpy(tot_images)
        
        print("Data loaded.")
        return tot_images
        

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        image = self.img_tensor[idx, :, :, :]
        label = self.img_labels[idx, : ]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # Restituiamo il tensore di una sola immagine e di una sola label
        return image, label


class SiameseNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Batch Normalization layer 1
        self.bn1 = nn.BatchNorm2d(num_features=1)
        
        # Batch Normalization layer 2
        self.bn2 = nn.BatchNorm1d(num_features=13456)
        
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5))
        
        # Average pooling layer
        self.avgp2_2 = nn.AvgPool2d(kernel_size=(2, 2))
        
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5))
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=13456, out_features=384)
        
        
    def forward(self, x):
        # bs: batch_size
        
        # Input: tensore di immagini bsx1x128x128 (1 è il canale perché siamo in scala di grigio)
        out = self.bn1(x)
        # Output: immagine bsx1x128x128 ma con valori normalizzati
        
        # Input: immagine bsx1x128x128 ma con valori normalizzati
        out = nn.functional.tanh(self.conv1(out))
        # Output: tensore bsx4x124x124
        
        # Input: tensore bsx4x124x124
        out = self.avgp2_2(out)
        # Output: bsx4x62x62
        
        # Input: bsx4x62x62
        out = nn.functional.tanh(self.conv2(out))
        # Output: bsx16x58x58
        
        # Input: bsx16x58x58
        out = self.avgp2_2(out)
        # Output: bsx16x29x29
        
        # Input: bsx16x29x29
        out = nn.Flatten()(out)
        # Output: tensore bsx13456
        
        # Input: tensore bsx13456
        out = self.bn2(out)
        # Output: bsx13456
        
        # Input: tensore bsx13456
        out = nn.functional.tanh(self.fc(out))
        # Output: tensore di bsx384 elementi della codifica frattale
        
        return out


class CustomLoss(torch.nn.Module):

    def __init__(self, distance):
        super().__init__()
        self.distance = distance
        
    def forward(self, output, label):
        batch_distances = np.zeros(shape=(16,), dtype=np.float32)
        
        # Calcola la distanza tra embedding previsto dalla rete ed embedding reale per tutte gli elementi della batch
        for i in range(output.shape[0]):
            temp = self.distance(output[i], label[i])
            batch_distances[i] = temp

        #print(f"loss: {loss}")
        #print(f"loss.shape: {loss.shape}")
        #print(f"type(loss): {type(loss)}")
        
        loss = torch.tensor(batch_distances, dtype=torch.float32, device=output.device)
        loss = loss.mean()
        #print(f"loss: {loss}")
        #print(f"loss.shape: {loss.shape}")        
        return loss


def custom_euclidean_distance(u, v):
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    return euclidean(u, v)


def custom_manhattan_distance(u, v):
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    return cityblock(u, v)


def custom_seuclidean_distance(u, v):
    # Calcolo della matrice di varianza e covarianza tra i due input
    mat = torch.cat((u, v))
    cov_mat = torch.cov(mat)
    
    # Prendo la diagonale
    diag = torch.diagonal(cov_mat)

    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    diag = diag.cpu().detach().numpy().flatten()
    
    return seuclidean(u, v, diag)
    

def custom_canberra_distance(u, v):
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    return canberra(u, v)


def custom_mahalanobis_distance(u, v):
    # Calcolo della matrice V^-1, l'inversa della matrice di varianza e covarianza tra i due input
    mat = torch.cat((u, v))
    cov_mat = torch.cov(mat)
    cov_mat_inv = torch.inverse(cov_mat)
    
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    cov_mat_inv = cov_mat_inv.cpu().detach().numpy()
    
    return mahalanobis(u, v, cov_mat_inv)


def custom_chebyshev_distance(u, v):
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    return chebyshev(u, v)


def custom_braycurtis_distance(u, v):
    # Trasformazione degli oggetti in ndarray per poter usare la funzione di scipy.spatial.distance    
    u = u.cpu().detach().numpy().flatten()
    v = v.cpu().detach().numpy().flatten()
    
    return braycurtis(u, v)