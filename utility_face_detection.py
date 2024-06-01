
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

from network_files import my_utility as mu


def max_area(faces):
    max_area = 0
    for f in faces:
        (x1, y1, w1, h1) = f
        area = w1 * h1
        if(area > max_area):
            max_area = area
            (x, y, w, h) = f

    return (x, y, w, h)

def face_detection(image_path):
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.05, minNeighbors=12, minSize=(90, 90), maxSize=(150, 150)
    )
    if len(faces) > 1:
        x, y, w, h = max_area(faces)
    else:
        x, y, w, h = faces[0]
                    
    img_crop = gray_image[y:y+h, x:x+w]
    
    return img_crop
    

class TrainDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        
        # Importazione delle immagini
        self.img_tensor = self._createImgTensor()   
        
    
    def _createImgTensor(self):
        list_images = list()
        for file_name in os.listdir(self.img_dir):
            if not file_name.startswith('.'):
                # Carica l'immagine
                img = cv2.imread(os.path.join(self.img_dir, file_name))

                # Modifica la dimensione
                img = cv2.resize(img, (128, 128))

                img = mu.rgb2gray(img)
                
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
        print(tot_images.shape)
        print(len(tot_images))
        return tot_images
        

    def __len__(self):
        return len(self.img_tensor)
    
    def __getitem__(self, idx):
        image = self.img_tensor[idx, :, :, :]
        # Restituiamo il tensore di una sola immagine
        return image
    