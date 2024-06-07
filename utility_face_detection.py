
import numpy as np
import pandas as pd
import random
import shutil
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from network_files import my_utility as mu


def max_area(faces):
    max_area = 0
    for f in faces:
        (_, _, w1, h1) = f
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
        
        #print("Data loaded.")
        #print(tot_images.shape)
        #print(len(tot_images))
        return tot_images
        

    def __len__(self):
        return len(self.img_tensor)
    
    def __getitem__(self, idx):
        image = self.img_tensor[idx, :, :, :]
        # Restituiamo il tensore di una sola immagine
        return image


def split_data(user_path, train_ratio=0.8):
    train_path = os.path.join(user_path, "dataset_splitted\\train")
    test_path = os.path.join(user_path, "dataset_splitted\\test")
    
    subjects_path = os.path.join(user_path, "face_images")
    
    for label in os.listdir(subjects_path):
        label_path = os.path.join(subjects_path, label)
        
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            random.seed(42)
            random.shuffle(images)
            
            train_size = int(len(images) * train_ratio)
            
            train_images = images[:train_size]
            test_images = images[train_size:]
            
            train_label_path = os.path.join(train_path, label)
            test_label_path = os.path.join(test_path, label)
            
            if not os.path.exists(train_label_path):
                os.makedirs(train_label_path)
            if not os.path.exists(test_label_path):
                os.makedirs(test_label_path)
            
            for image in train_images:
                image_path = os.path.join(label_path, image)
                image_crop = face_detection(image_path)
                dst = os.path.join(train_label_path, image[:11])
                cv2.imwrite(dst + '.jpg', image_crop)
            
            for image in test_images:
                src = os.path.join(label_path, image)
                dst = os.path.join(test_label_path, image)
                shutil.copy(src, dst)
    
    print("Splitting eseguito!")


def create_train_template(user_path):
    train_path = os.path.join(user_path, "dataset_splitted\\train")
    
    model = mu.SiameseNeuralNetwork()
    model_path = torch.load(f="model_all_images")
    model.load_state_dict(model_path)
    model.eval()
    
    codify_list = list()    
    labels_list = list()
    
    for i in range(1,25):
        label = str(i).rjust(2,'0')
        train_curr_images = os.path.join(train_path, label)
        imagesDataSet = TrainDataset(img_dir=train_curr_images)
        img_dataloader = DataLoader(imagesDataSet, batch_size=1, drop_last=True)
        
        for image in img_dataloader:
            codify_list.append(model(image).cpu().detach().numpy().flatten())
            labels_list.append(i)
            
    codify_df = pd.DataFrame(codify_list)
    codify_df.insert(384, "Label", labels_list)
    
    codify_df.to_csv("template.csv", sep=",", index = False)