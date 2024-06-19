# Evaluating Distance Functions in Fractal-Encoded Siamese Networks


## Abstract
In questo lavoro è stato presentato uno studio sull'applicazione delle reti siamesi per l'emulazione della **codifica frattale** ai compiti di riconoscimento facciale, con particolare focus sulla valutazione delle **funzioni di distanza** utilizzate come metro di confronto tra risultati. 

L'approccio prevede la pre-elaborazione di due dataset ampiamente utilizzati, *CelebA* e *Biwi Kinect* Head Pose Database, per addestrare e testare la rete. 

È stato analizzato l'impatto della variazione del numero di immagini modello per soggetto sull'**accuratezza** della classificazione dimostrando un progressivo miglioramento delle prestazioni di classificazione all'aumentare delle immagini. 

Questi risultati sottolineano l'importanza della funzione di distanza utilizzata ma anche del dataset utilizzato per task di face recognition.

## Prerequisiti

1. Creare un ambiente virtuale conda utilizzando il seguente comando nel terminale:
```conda create my_env```
2. Python 3.10.4 (per permettere alle librerie di essere installate correttamente).
3. Installare le librerie in `requirements.txt` utilizzando il seguente comando nel terminale:
```pip install -r requirements.txt```
4. Scarica il dataset **Biwi Kinect** a questo [link](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database?select=faces_0) facendo attenzione a scaricare solo la cartella "*faces_0*".


## Data Cleaning
- Una volta scaricato il dataset si raccomanda di eliminare in "*faces_0*" tutti file che non sono in formato `.png`, questo per ogni soggetto.

- Per contenere le cartelle che verranno create all'esecuzione del test, si raccomanda di creare una cartella nel file system chiamata "*faceDetection*" inserendo al suo interno "*faces_0*".



## Script and Folder Description
Nella repository sono presenti varie cartelle con le varie strutture e test utilizzati per la sperimentazione:
- 📂**confronto:** qui ci è presente un esempio di test per visualizzare varie distanze di Canberra utilizzando come esempio due immagini simili;
- 📂**face_detection:** qui è presente il codice che filtra le immagini su cui viene riconosciuto un volto;
- 📂**network_files:** qui sono presenti i codici in PyTorch della rete siamese utilizzata nei test;
- 📂**more_testing:** qui sono presenti i codici che sono stati utilizzati per visualizzare il comportamento della classificazione aumentando progressivamente le immagini di train.

Per eseguire il test della sperimentazione bisogna eseguire prima `train_template.py`, stando attenti a sostituire nella variabile `user_path` il percorso per accedere alla cartella "*faceDetection*".

Dopodichè bisogna eseguire `test_realtime.py`, stando attenti a sostituire nella variabile `user_path` il percorso per accedere alla cartella "*faceDetection*".

Verrà prodotto un file chiamato `results.csv` dove si possono visualizzare i risultati del test.