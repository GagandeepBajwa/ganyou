'''
File Description: load the data from the directory
'''
import os
import sys
import numpy as np
from PIL import Image

def read_image(image):
    print("Reading the given image file")
    im = Image.open(image)
    image_array = np.array(im, dtype=np.float64)
    image_array = np.nan_to_num(image_array)
    print('Shape of the iamge that has been read: ', image_array.shape, ' and unique values: ', np.unique(image_array))
    #return the numpy array
    return image_array

def load_data(dir):
    print("Starting loading data: ", dir)
    data=[]
    for file in os.listdir(dir):
        #read the file and push to the files 
        image_array = read_image(os.path.join(dir,file))
        data.append(image_array)
    
    return data

if __name__ ==  "__main__":
    print("Starting Loading the data")