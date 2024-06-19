import numpy as np
import pandas as pd
import os
import cv2
import torch
from network_files import my_utility as mu
from utility_face_detection import face_detection
from scipy.spatial.distance import canberra

def image_test(image_path, model, template, labels):
    img_crop = face_detection(image_path)
    img_crop = cv2.resize(img_crop, (128, 128))

    # Converte in np.float32
    img_crop = np.asarray(img_crop, dtype=np.float32)
    
    # Converto l'immagine in tensore
    img_tensor = torch.from_numpy(img_crop)
    
    # Aggiungo le dimensioni relative a canale e batch (1)
    img_tensor = torch.reshape(img_tensor, (1, 1, 128, 128))
    
    # Calcolo della codifica dell'immagine e converto in numpy array
    codify = model(img_tensor).cpu().detach().numpy().flatten()
    
    # Matching
    min_dist = np.inf
    min_label = None

    for index, row in template.iterrows():
        row_array = row.to_numpy()
        distance = canberra(codify, row_array)
        
        if distance < min_dist:
            min_dist = distance
            min_label = labels.loc[index]
        
    
    return min_label


def test_matching(test_path, template_path):
    # Caricamento del modello
    model = mu.SiameseNeuralNetwork()
    model_path = torch.load(f="model_all_images")
    model.load_state_dict(model_path)
    model.eval()
    
    # Caricamento e manipolazione template e label
    template = pd.read_csv(template_path)
    
    true_labels_df = template["Label"]
    template_df = template.drop("Label", axis=1)
    
    out_true_labels = list()
    out_pred_labels = list()
    
    # Per ogni immagine di test, fa il matching
    for label in os.listdir(test_path):
        src = os.path.join(test_path, label)
        
        for file_name in os.listdir(src):
            if not file_name.startswith('.'):
                test_image_path = os.path.join(src, file_name)
        
                print(f"test_image_path: {test_image_path}")
                pred_label = image_test(test_image_path, model, template_df, true_labels_df)
                out_pred_labels.append(pred_label)
                out_true_labels.append(label)
    
    return out_true_labels, out_pred_labels




# SCRIVERE PATH FINO ALLA CARTELLA CONTENENTE LE IMMAGINI (FUORI)
user_path = "C:\\Users\\User\\Progetti_universita\\Progetto-FVAB"

test_path = os.path.join(user_path, "faceDetection\\dataset_splitted\\test")
template_path = os.path.join(user_path, "template.csv")

true_labels, pred_labels = test_matching(test_path, template_path)

true_labels_int = [int(i) for i in true_labels]
pred_labels_int = [int(i) for i in pred_labels]

results = pd.DataFrame({"True labels": true_labels_int,
                        "Pred labels": pred_labels_int})

results.to_csv("results.csv", sep=",", index = False)