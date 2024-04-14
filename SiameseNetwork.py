import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import my_utility as mu
from keras.optimizers.schedules import ExponentialDecay
import keras.ops as K
from keras.losses import MeanSquaredError

tf.config.run_functions_eagerly(True)
#parametri della rete
epochs = 10
batch_size = 16
margin = 1  # Margin for contrastive loss.


n=384 #dimensione codifica frattale

#parametri del dataset
num_samples_train=8000
num_samples_val=2000
num_samples_test=2000


#inizializzazione
target_distances = np.zeros(num_samples_train)
val_target_distances=np.zeros(num_samples_val)

#insalle cartelle dei file
dir_codify=r'C:\Users\antho\Desktop\img_celeba_10000'
dir_images_train=r'C:\Users\antho\Desktop\img_celeba_10000\img_celeba_10000_train'
dir_images_val=r'C:\Users\antho\Desktop\img_celeba_10000\img_celeba_10000_valid'
dir_image_test=r'C:\Users\antho\Desktop\img_celeba_10000\img_celeba_2000_test'

#path di immagini ed embeddings
train_embeddings=mu.open_csv(dir_codify,'\codify_celeba_10000_train.csv',num_samples_train)
val_embeddings=mu.open_csv(dir_codify,'\codify_celeba_10000_valid.csv',num_samples_val)
test_embeddings=mu.open_csv(dir_codify,'\codify_celeba_2000_test.csv',num_samples_test)
train_images=mu.image_loader(dir_images_train)
val_images=mu.image_loader(dir_images_val)
test_images=mu.image_loader(dir_image_test)



#definizione degli input della siamese
input_image = keras.layers.Input((128, 128, 1))
input_embedding = keras.layers.Input((384,))

#ramo CNN della siamese per l'immagine
x = keras.layers.BatchNormalization()(input_image)
x = keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)

x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(n, activation="tanh")(x)
embedding_network = keras.Model(input_image, x)

#chiamo la rete che genera l'embedding sull'immagine in input
tower_image = embedding_network(input_image)

#merge dell'Embedding generato dalla CNN con l'embedding in input
merge_layer = keras.layers.Lambda(mu.euclidean_distance, output_shape=(1,))(
    [tower_image, input_embedding]
)

#merge layer
siamese = keras.Model(inputs=[input_image, input_embedding], outputs=merge_layer)

#learning rate personalizzato (opzionale)
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=0.01)



#chiamata alla loss personalizzata in my utility
siamese.compile(optimizer=optimizer, loss=mu.loss(margin=margin))
#chiamata ad una loss di keras
#siamese.compile(optimizer=optimizer, loss=MeanSquaredError)
siamese.summary()



# Verifica e conversione dei dati per la rete
train_images = np.array(train_images, dtype=np.float32)
train_embeddings = np.array(train_embeddings, dtype=np.float32)

val_images = np.array(val_images, dtype=np.float32)
val_embeddings = np.array(val_embeddings, dtype=np.float32)

test_images = np.array(val_images, dtype=np.float32)
test_embeddings = np.array(val_embeddings, dtype=np.float32)

# Assicurati che le etichette abbiano la stessa lunghezza del numero di campioni di addestramento
assert len(train_embeddings) == num_samples_train, "Errore: Lunghezza delle etichette non corrispondente al numero di campioni di addestramento."

#addestramento
history = siamese.fit([train_images, train_embeddings], train_embeddings, epochs=epochs, batch_size=batch_size,
                      validation_data=([val_images, val_embeddings], val_embeddings))



# Valutazione della rete
eval_result = siamese.evaluate([test_images, test_embeddings], test_embeddings)

# Calcolo della distanza Euclidea per un esempio di test
example_index = 0
test_image = np.expand_dims(test_images[example_index], axis=0)
test_embedding = np.expand_dims(test_embeddings[example_index], axis=0)
predicted_embedding = siamese.predict([test_image, test_embedding])

# Calcolo della distanza Euclidea in numpy
euclidean_distance = np.linalg.norm(test_embedding - predicted_embedding)

print(f"Distanza Euclidea per l'immagine di test: {euclidean_distance}")

# Visualizzazione della curva di apprendimento
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()