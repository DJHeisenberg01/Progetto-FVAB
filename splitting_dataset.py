import os
import shutil
import random

def split_data(source_dir, train_dir, test_dir, train_ratio=0.8):
    
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
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
                src = os.path.join(category_path, image)
                dst = os.path.join(train_category_dir, image)
                shutil.copy(src, dst)
            
            for image in test_images:
                src = os.path.join(category_path, image)
                dst = os.path.join(test_category_dir, image)
                shutil.copy(src, dst)
    
    print("Splitting eseguito!")

source_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\faces_0'

train_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\dataset_splitted\\train'
test_directory = 'C:\\Users\\antho\\Desktop\\faceDetection\\dataset_splitted\\test'

split_data(source_directory, train_directory, test_directory)
