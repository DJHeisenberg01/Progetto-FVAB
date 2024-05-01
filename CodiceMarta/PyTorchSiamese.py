import csv
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import my_utility as mu
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset
from scipy.spatial import distance 
from sklearn.metrics import pairwise
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm


# parametri della rete
epochs = 10
batch_size = 16
margin = 1  # Margin for contrastive loss.


n=384 # dimensione codifica frattale

# parametri del dataset
num_samples_train=8000
num_samples_val=2000
num_samples_test=2000

# inizializzazione
target_distances = np.zeros(num_samples_train)
val_target_distances=np.zeros(num_samples_val)

# aggiungere path locale
current_path=str(os.getcwd())

# cartelle dei file
dir_codify = current_path +'\img_celeba_10000'
dir_codify_train= current_path + '\img_celeba_10000\codify_celeba_10000_train'
dir_codify_val= current_path +'\img_celeba_10000\codify_celeba_10000_valid'
dir_codify_test= current_path +'\img_celeba_10000\codify_celeba_2000_test'
dir_images_train= current_path +'\img_celeba_10000\img_celeba_10000_train'
dir_images_val= current_path +'\img_celeba_10000\img_celeba_10000_valid'
dir_images_test= current_path +'\img_celeba_10000\img_celeba_2000_test'
# cartelle dei file
dir_codify = current_path + r'\img_celeba_10000'
csv_train = dir_codify + r'\codify_celeba_10000_train.csv'
csv_val = dir_codify + r'\codify_celeba_10000_valid.csv'
csv_test = dir_codify + r'\codify_celeba_2000_test.csv'

# path di immagini ed embeddings
train_embeddings = mu.open_csv(csv_train, num_samples_train)
val_embeddings = mu.open_csv(csv_val, num_samples_val)
test_embeddings = mu.open_csv(csv_test, num_samples_test)


# creazione dataset per l'apertura delle immagini
class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform,):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


transform_image = transforms.Compose([
    transforms.Grayscale(),  # Converti in scala di grigio
    transforms.Resize((128, 128)),  # Ridimensiona all'input della tua rete Siamese
    transforms.ToTensor()  # Converti l'immagine in tensore
])


train_images = CustomImageDataset(dir_images_train, transform=transform_image)
test_images = CustomImageDataset(dir_images_test, transform=transform_image)


"""
    Il DataLoader gestisce la divisione del dataset in batch, per migliorare l'efficienza 
    e la precisione nella fase training. Il parametro shuffle se impostato a True rimescola 
    i dati in ogni epoca. Tale funzionalità è utile per il training, ma non per il test.
"""
train_image_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
train_embeddings_loader = DataLoader(train_embeddings, batch_size=batch_size, shuffle=True)
test_image_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)
test_embeddings_loader = DataLoader(test_embeddings, batch_size=batch_size, shuffle=False)



"""
    In Keras l'image_input viene passata durante la definizione della rete.
    Per migliorare la leggibilità del codice sono stati definiti separatamente 
    i livelli della rete rispetto all'esecuzione della stessa (vedi metodo forward)
"""

print("prima del training")
# Architettura della CNN Siamese
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(1),            # 1 perchè consideriamo solo un canale
            nn.Conv2d(1, 4, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),        # kernel_size = pool_size
            nn.Conv2d(4, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),        # kernel_size = pool_size
            nn.Flatten(),
            nn.Linear(16*29*29, 384),                     # n = 384 della codifica frattale
            nn.Tanh()
        )
        

    def forward(self, image):
        # Il risultato sarà l'embedding della rete
        image = image.float()
        output = self.network(image)
        return output
    
    """
    def forward(self, image, embedding):
        # L'output è la distanza calcolata dal layer di merge 
        image = image.float()
        #image = image.unsqueeze(1)
        output_embedding = self.network(image)
        return merge_layer(output_embedding, embedding)
    """


# Definizione del device 
device = torch.device('cpu')
model = SiameseNetwork().to(device)

# Definizione della funzione di perdita e dell'algoritmo di ottimizzazione
criterion = mu.Loss(margin=margin)
#criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#print(merge(train_embeddings[1], train_embeddings[1]))
#print(merge(train_embeddings[1], train_embeddings[10]))


def training(model, device,  num_epochs, train_image_loader, train_embeddings_loader):
    n_total_steps = len(train_image_loader)
    counter = []
    loss_history = [] 
    iteration_number= 0
    # itera per un numero di epoche prefissato
    for epoch in range(num_epochs):

        # itera sui branch del training dei data loader
        for i, (images, embeddings) in enumerate(zip(train_image_loader, train_embeddings_loader)):
            # sposta i dati sul device
            images = images.to(device)
            embeddings = embeddings.to(device)

            # l'output predetto dal modello
            outputs = model(images)

            # calcolo funzione di perdita tra il risultato vero e la predizione
            loss = criterion(embeddings, outputs)
        
            loss.requires_grad = True
            # ottimizzazione
            optimizer.zero_grad() # azzera i gradienti, necessario prima di calcolare i gradienti nella fase di backpropagation
            loss.backward()       # fase di backpropagation, calcola i gradienti della funzione di perdita rispetto ai parametri del modello
            optimizer.step()      # ottimizzatione, si aggiornano i parametri del modello utilizzando gli gradienti calcolati in precedenza
            
            iteration_number += 1
            counter.append(iteration_number)
            loss_history.append(loss.item())

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            


    print("Fine del training")
    plt.plot(counter, loss_history)
    plt.show()





training(model, device, epochs, train_image_loader, train_embeddings_loader)


def evaluation(model, device, test_image_loader, test_embeddings_loader):
    model.eval()
    counter = []
    loss_history = [] 
    iteration_number= 0
    n_total_steps = len(test_image_loader)

    with torch.no_grad():   # senza tener traccia dei gradienti
        for i, (images, embeddings) in enumerate(zip(test_image_loader, test_embeddings_loader)):
            images = images.to(device)
            embeddings = embeddings.to(device)

            # predizione del modello
            outputs = model(images)
            loss = criterion(embeddings, outputs)
            iteration_number += 1
            counter.append(iteration_number)
            loss_history.append(loss.item())
            if (i+1) % 10 == 0:
                print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    plt.plot(counter, loss_history)
    plt.show()


evaluation(model, device, test_image_loader, test_embeddings_loader)


# Calcolo della distanza per un esempio di test
example_index = 0
image = test_images[example_index].unsqueeze(1)
embedding = test_embeddings[example_index]
predicted_embedding = model(image)


# Calcolo della distanza
distance = mu.merge_layer(predicted_embedding, embedding)

print(f"Distanza per l'immagine di test: {distance}")
