from faceDetection.face_detection import discard_no_faces
from utility_face_detection import split_data, create_train_template

# SPECIFICARE PATH FINO ALLA CARTELLA PRIMA DI QUELLA CONTENENTE LE IMMAGINI 
user_path = "C:\\Users\\User\\Progetti_universita\\Progetto-FVAB\\faceDetection"

#crea una nuova cartella chiamata face_images con le immagini filtrate
discard_no_faces(user_path)

#attua lo split del dataset 80% train e 20% test
split_data(user_path)

#estrae i volti e crea un file template.csv con tutte le codifiche dei soggetti del train set
create_train_template(user_path)

