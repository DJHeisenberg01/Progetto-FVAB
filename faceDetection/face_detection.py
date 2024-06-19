import cv2
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Funzione che, tra le facce restituite dal face detector, seleziona solo quella con la finestra più grande
def max_area(faces):
    max_area = 0
    for f in faces:
        (_, _, w1, h1) = f
        area = w1 * h1
        if(area > max_area):
            max_area = area
            (x, y, w, h) = f

    return (x, y, w, h)

#Estrae dal dataset originale face_0 solo le immagini su cui viene riconosciuto un volto dal face detector
def discard_no_faces(user_path):
    image_folder = "faces_0\\"

    for i in range(1, 25):
        count_acquisite = 0
        count_scartate = 0
        
        label = str(i).rjust(2,'0')
        write_path = os.path.join(user_path, "face_images", label)
        
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        
        read_path = os.path.join(user_path, image_folder, label)
        
        for file_name in os.listdir(read_path):
            if not file_name.startswith('.'):
                image_path = os.path.join(read_path, file_name)
                img = cv2.imread(image_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                try:
                    faces = face_classifier.detectMultiScale(
                        gray_image, scaleFactor=1.05, minNeighbors=12, minSize=(90, 90), maxSize=(150, 150)
                    )  
                    if len(faces) > 1:
                        x, y, w, h = max_area(faces)
                    else:
                        x, y, w, h = faces[0]
                    
                    img_crop = gray_image[y:y+h, x:x+w]
                    #img_crop = cv2.resize(img_crop, (128, 128))

                    cv2.imwrite(os.path.join(write_path, file_name), img)
                    
                    count_acquisite += 1

                except:
                    count_scartate += 1

        