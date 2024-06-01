import os
import shutil
import random
import cv2
from utility_face_detection import face_detection, TrainDataset
import csv
from torch.utils.data import DataLoader

def split_data(source_dir, train_dir, test_dir, train_ratio=0.8):
    
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            random.seed(42)
            random.shuffle(images)
            
            train_size = int(len(images) * train_ratio)
            
            train_images = images[:train_size]
            test_images = images[train_size:]
            
            train_category_dir = os.path.join(train_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            
            if not os.path.exists(train_category_dir):
                os.makedirs(train_category_dir)
            if not os.path.exists(test_category_dir):
                os.makedirs(test_category_dir)
            
            for image in train_images:
                image_crop = face_detection(category_path + "\\" + image)
                dst = os.path.join(train_category_dir, image[:11])
                cv2.imwrite(dst  + '.jpg', image_crop)
            
            for image in test_images:
                src = os.path.join(category_path, image)
                dst = os.path.join(test_category_dir, image)
                shutil.copy(src, dst)
    
    print("Splitting eseguito!")
    
def codify_train_images(train_images):
    images = TrainDataset(img_dir=train_images)
    img_dataloader = DataLoader(images, batch_size=1, drop_last=True)
            

source_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\face_images'

train_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\dataset_splitted\\train'
test_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\dataset_splitted\\test'

split_data(source_directory, train_directory, test_directory)
