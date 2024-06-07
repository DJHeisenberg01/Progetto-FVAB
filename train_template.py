from faceDetection.face_detection import discard_no_faces
from utility_face_detection import split_data, create_train_template

# SPECIFICARE PATH FINO ALLA CARTELLA PRIMA DI QUELLA CONTENENTE LE IMMAGINI 
user_path = "C:\\Users\\User\\Progetti_universita\\Progetto-FVAB\\faceDetection"

discard_no_faces(user_path)

split_data(user_path)

create_train_template(user_path)

