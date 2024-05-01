import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import csv
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import torch
from scipy.spatial import distance 


def open_csv(dir,n):
    with open(dir) as csv_file:
        dataread = list(csv.reader(csv_file, delimiter=','))
        fulldata = [x for x in dataread if x != []]
        arr_created = np.asarray(fulldata)
        shape = (n, 384)  # the array size depend on the dimension of the codec
        data_ar = arr_created.reshape(shape)            
        float_array = data_ar.astype(np.float32)
        #data = torch.tensor(float_array)
        data = torch.from_numpy(float_array)
        return data


def image_loader(dir_images):#,dir_codify):

    list_images=[]
    count=0
    for file_name in os.listdir(dir_images):
        if not file_name.startswith('.'):
            count=count+1
            codify_name=file_name + '.csv'
            img = cv2.imread(os.path.join(dir_images, file_name))
            img = cv2.resize(img, (128, 128))
            img = rgb2gray(img)
            img_array=np.asarray(img)
            list_images.append(img_array)

    tot_images = np.asarray(list_images)
    print(tot_images.shape)
    print(type(tot_images))
    tot_images = tot_images.reshape(count, 128,128)
    data = torch.from_numpy(tot_images, requires_grad=True)
    return data


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

"""
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))


def similarity_accuracy(y_true, y_pred, threshold=0.5):
    return tf.keras.metrics.binary_accuracy(tf.cast(y_true < threshold, tf.float32), tf.cast(y_pred < threshold, tf.float32))

def loss(margin=1):
    Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.

        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss


"""

""" def loss(margin=1):
    def contrastive_loss(y_true, y_pred):

        square_pred = torch.pow(y_pred)
        margin_square = torch.pow(torch.max(margin - (y_pred), 0))
        return torch.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss
"""

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        """
            Al posto di :
                square_pred = ops.square(y_pred)
                margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
                return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

            Per ulteriori informazioni su questo calcolo vedi la lezione sul Deep Learning
        """

        square_pred = np.power(y_pred, 2)
        margin_square = np.power(np.maximum(margin - y_pred, 0), 2)
        return np.mean((1 - y_true) * square_pred + y_true * margin_square)

    return contrastive_loss


def merge_layer(image, embeddings):
    #x = image.detach().numpy().reshape(-1, 1)
    #y = embeddings.detach().numpy().reshape(-1, 1)
    x = image.detach().numpy().reshape(-1)  # Reshape image to 1-D array
    y = embeddings.detach().numpy().reshape(-1)  # Reshape embeddings to 1-D array
    return distance.canberra(x, y)
    # distance = nn.PairwiseDistance()
    # return distance(image, embeddings)

    # return pairwise.cosine_distances(x, y)


class Loss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(Loss, self).__init__()
        self.margin = margin

    def forward(self, y_true, y_pred):
        # Versione con la liberia
        # Converti i tensori PyTorch in numpy array
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

        # Calcola la distanza di Canberra tra y_true e y_pred
        distances = distance.canberra(y_true_np, y_pred_np)

        # Convert the distances to a PyTorch tensor
        loss = torch.tensor(distances.mean(), dtype=torch.float32, device=y_true.device)

        return loss
        """
        # Versione Manuale
        # Calcola la distanza di Canberra tra y_true e y_pred
        distance = torch.sum(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred) + 1e-8), dim=1)

        # Calcola la loss usando una funzione basata sulla distanza di Canberra
        loss = torch.mean(distance)
        return loss
        """