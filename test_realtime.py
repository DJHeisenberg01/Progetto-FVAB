import numpy as np
import pandas as pd
import os
import cv2
import torch
from network_files import my_utility as mu
from utility_face_detection import face_detection
from scipy.spatial.distance import canberra

# Sulla singola immagine di test
def image_test(image_path, model, template, labels):
    # Si individua il volto dell'individuo
    img_crop = face_detection(image_path)
    
    # Se ne fa il resize, in modo da poter dare l'immagine in input alla rete
    img_crop = cv2.resize(img_crop, (128, 128))

    # Conversione in np.float32
    img_crop = np.asarray(img_crop, dtype=np.float32)
    
    # Conversione dell'immagine in tensore
    img_tensor = torch.from_numpy(img_crop)
    
    # Aggiungo le dimensioni relative a canale e batch (1)
    img_tensor = torch.reshape(img_tensor, (1, 1, 128, 128))
    
    # Calcolo della codifica dell'immagine e converto in numpy array
    codify = model(img_tensor).cpu().detach().numpy().flatten()
    
    ## Matching
    # Inizializzo le variabili relative al miglior risultato
    min_dist = np.inf
    min_label = None

    # Per ogni codifica nel template di training
    for index, row in template.iterrows():
        # Converto la riga in numpy array
        row_array = row.to_numpy()
        
        # Calcolo la distanza di Canberra tra la codifica dell'immagine ricevuta in input e l'attuale codifica di training
        distance = canberra(codify, row_array)
        
        # Se la distanza tra queste due codifiche è minore della più piccola individuata finora
        if distance < min_dist:
            # Salva la distanza
            min_dist = distance
            # E la label prevista per codifica
            min_label = labels.loc[index]
        
    # Si restituisce la label prevista
    return min_label


def test_matching(test_path, template_path):
    ## Istanziamento del modello
    model = mu.SiameseNeuralNetwork()
    model_path = torch.load(f="model_all_images")
    
    # Caricamento dei pesi del modello precedentemente addestrato
    model.load_state_dict(model_path)
    model.eval()
    
    ## Caricamento e manipolazione template e label
    template = pd.read_csv(template_path)
    
    true_labels_df = template["Label"]
    template_df = template.drop("Label", axis=1)
    
    # Liste di true labels e pred labels da confrontare
    out_true_labels = list()
    out_pred_labels = list()
    
    # Per ogni cartella di ogni individuo
    for label in os.listdir(test_path):
        # Crea il path per quella cartella
        src = os.path.join(test_path, label)
        
        # Per ogni immagine della cartella di quell'indiviuo
        for file_name in os.listdir(src):
            if not file_name.startswith('.'):
                # Crea il path per quell'immagine
                test_image_path = os.path.join(src, file_name)
        
                print(f"test_image_path: {test_image_path}")
                
                # Prevede la label per l'immagine
                pred_label = image_test(test_image_path, model, template_df, true_labels_df)
                
                # Salva i valori della true label e della pred label
                out_pred_labels.append(pred_label)
                out_true_labels.append(label)
    
    return out_true_labels, out_pred_labels




# SCRIVERE PATH FINO ALLA CARTELLA CONTENENTE LE IMMAGINI (FUORI)
user_path = "C:\\Users\\User\\Progetti_universita\\Progetto-FVAB"

test_path = os.path.join(user_path, "faceDetection\\dataset_splitted\\test")
template_path = os.path.join(user_path, "template.csv")

# Avvio del matching tra le immagini di testing ed il template di training
true_labels, pred_labels = test_matching(test_path, template_path)

# Conversione dei risultati da stringhe ad interi
true_labels_int = [int(i) for i in true_labels]
pred_labels_int = [int(i) for i in pred_labels]

# Creazione del dizionario di risultati
results = pd.DataFrame({"True labels": true_labels_int,
                        "Pred labels": pred_labels_int})

# Scrittura del dizionario in un .csv
results.to_csv("results.csv", sep=",", index = False)