import matplotlib.image as mpimg
from PIL import Image
from scipy import ndimage
import numpy as np
#np.warnings.filterwarnings('ignore')
import csv
import os
import cv2


#Get GreyScale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# Transformations

def reduce(img, factor):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
    return result

def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
    return img[::direction, :]

def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast * rotate(flip(img, direction), angle) + brightness

# Contrast and brightness

def find_contrast_and_brightness1(D, S):
    #Fix the contrast and only fit the brightness
    contrast = 0.75
    brightness = (np.sum(D - contrast * S)) / D.size
    return contrast, brightness

def find_contrast_and_brightness2(D, S):
    #Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]

# Compression
def generate_all_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):

            # Extract the source block and reduce it to the shape of a destination block
            S = reduce(img[k * step:k * step + source_size, l * step:l * step + source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append((k, l, direction, angle, apply_transformation(S, direction, angle)))
    return transformed_blocks

def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            #print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i * destination_size:(i + 1) * destination_size, j * destination_size:(j + 1) * destination_size]
            # Test all possible transformations and take the best one
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness2(D, S)
                S = contrast * S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
                    #transformations[i][j] = (k, l, direction, angle)
    return transformations

# Parameters

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]

def write_list_to_file(guest_list, filename):
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ';')
        for entries in guest_list:
            csvwriter.writerows(entries) #transf_blocks-->writerow
    csvfile.close()

# Tests

def test_greyscale():

    #Directories
    directory_in = "C:\\Users\\User\\Desktop\\prove"

    # ______________________________________________________________________
    # define the name of the directory to be created
    directory_csv = "C:\\Users\\User\\Desktop\\prove_out"

    try:
        os.mkdir(directory_csv)
    except OSError:
        print("Creation of the directory %s failed" % directory_csv)
    else:
        print("Successfully created the directory %s " % directory_csv)
    # ______________________________________________________________________

    #countImages

    count = 0

    #All Images
    for file_name in os.listdir(directory_in):
        if not file_name.startswith('.'):
            count = count + 1
            img = cv2.imread(os.path.join(directory_in, file_name))
            img = cv2.resize(img, (128,128))
            img = rgb2gray(img)
            img = reduce(img, 4)
            transformations = compress(img, 8, 4, 8) #8,4,8 - #16,8,16
            #write the CSV_Transformations
            write_list_to_file(transformations,(os.path.join(directory_csv, file_name + '.csv')))
            print(count)

    print("All images: %i" % count)

if __name__ == '__main__':
    test_greyscale()


