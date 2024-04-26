import numpy as np
<<<<<<< HEAD
import keras
from keras import ops
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
=======
from keras import ops # DA TOGLIERE
#import matplotlib.pyplot as plt
import tensorflow as tf # DA TOGLIERE
>>>>>>> 2e2d1d435bd9c1b3f87ddf13b29768619c4009b9
import os
import cv2
#import csv
#from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
<<<<<<< HEAD
from tensorflow.keras.callbacks import Callback
from scipy.spatial.distance import cityblock
import sklearn.metrics.pairwise as calc
=======
#from scipy.spatial.distance import cityblock

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
>>>>>>> 2e2d1d435bd9c1b3f87ddf13b29768619c4009b9


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


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output, label):
        square_pred = (1-label) * torch.pow(output, 2)
        margin_square = (label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2)
        loss_contrastive = torch.mean(square_pred + margin_square)
        return loss_contrastive

#def open_csv(dir,file_name,n):
#    with open(dir+file_name) as csv_file:
#        dataread = list(csv.reader(csv_file, delimiter=','))
#        fulldata = [x for x in dataread if x != []]
#        arr_created = np.asarray(fulldata, dtype=np.float32)
#        tensor_created = torch.from_numpy(arr_created)
#        shape = (n, 384)  # the array size depend on the dimension of the codec
#        data_ar = torch.reshape(tensor_created, shape)
#        return data_ar
#
#
#def image_loader(dir_images):#,dir_codify):
#    list_images=[]
#    count=0
#    for file_name in os.listdir(dir_images):
#        if not file_name.startswith('.'):
#            count=count+1
#            codify_name=file_name + '.csv'
#            img = cv2.imread(os.path.join(dir_images, file_name))
#            img = cv2.resize(img, (128, 128))
#            img = rgb2gray(img)
#            img_array=np.asarray(img, dtype=np.float32)
#            list_images.append(img_array)
#
#    tot_images = np.asarray(list_images)
#    tot_images = torch.from_numpy(tot_images)
#    print(tot_images.shape)
#    print(type(tot_images))   
#    return tot_images


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def manhattan_distance(vects):
    x, y = vects   
    return tf.reduce_sum(tf.abs(x - y), axis=1, keepdims=True)

def canberra_distance(vects):
    x, y = vects
    numerator = tf.abs(x - y)
    denominator = tf.abs(x) + tf.abs(y)
    return tf.reduce_sum(numerator / denominator, axis=1, keepdims=True)

def hamming_distance(vects):
    x, y = vects
    
    
    #return tf.reduce_mean(tf.cast(tf.not_equal(x1, y1), tf.float32), axis=-1, keepdims=True)
    distance = np.mean(np.not_equal(x, y))
    
    return distance

def chebyshev_distance(vects):
    x, y = vects
    chebyshev_diff = tf.abs(x - y)
    chebyshev_max = tf.reduce_max(chebyshev_diff, axis=1)
    return chebyshev_max

def bray_curtis_distance(vects):
    x, y = vects
    sum_diff = tf.reduce_sum(tf.abs(x - y), axis=1, keepdims=True)
    sum_sum = tf.reduce_sum(tf.abs(x + y), axis=1, keepdims=True)
    
    return tf.divide(sum_diff, sum_sum)


#covariance_matrix da definire
def mahalanobis_distance(vects, covariance_matrix):
    x, y = vects
    diff = x - y
    diff_transpose = tf.transpose(diff)
    mahalanobis = tf.linalg.matmul(diff, tf.linalg.inv(covariance_matrix))
    mahalanobis = tf.linalg.matmul(mahalanobis, diff_transpose)
    return tf.sqrt(mahalanobis)

def cosine_distance(vects):
    x, y = vects
    cos = tf.keras.losses.CosineSimilarity(axis=-1, reduction="sum_over_batch_size", name="cosine_similarity")
    return cos.call(x,y)

def standard_euclidean(vects):
    x, y = vects
    #manca il vettore delle varianze
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))



def similarity_accuracy(y_true, y_pred, threshold=0.5):
    return tf.keras.metrics.binary_accuracy(tf.cast(y_true < threshold, tf.float32), tf.cast(y_pred < threshold, tf.float32))

def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss


