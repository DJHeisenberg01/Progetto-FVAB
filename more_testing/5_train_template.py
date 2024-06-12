from face_detection import discard_no_faces
from utility_face_detection import split_data, create_train_template, face_detection
import cv2
import os

# SPECIFICARE PATH FINO ALLA CARTELLA PRIMA DI QUELLA CONTENENTE LE IMMAGINI 
user_path = "C:\\Users\\antho\\Desktop\\faceDetection"

#discard_no_faces(user_path)

#split_data(user_path)

#create_train_template(user_path)



'''image_train_path = os.path.join(user_path, "train_5_images")

for i in range(1, 25):
    label = str(i).rjust(2,'0')
    write_path = os.path.join(user_path, "train_5_faces", label)
    if not os.path.exists(write_path):
            os.makedirs(write_path)
        
    read_path = os.path.join(image_train_path, label)
    for file_name in os.listdir(read_path):
        if not file_name.startswith('.'):
            image_path = os.path.join(read_path, file_name)
            image_crop = face_detection(image_path)
            dst_path = os.path.join(write_path, file_name)
            cv2.imwrite(dst_path + '.jpg', image_crop)
            '''
            
create_train_template(user_path)