# This is a script for cropping the generated images
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from PIL import Image
import time
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

# Function from taken from the book to load all images from a directory
def img_loader(path, img_size=(128, 128)):
    temp = []
    for file in listdir(path):
        temp.append(img_to_array((load_img(path + "\\" + file, color_mode='grayscale'))))
    return np.array(temp)

# uncomment the needed part

path_to_normal_images = r"D:\Cyfronet\unpacked\4 batches\Normal"
# path_to_normal_images = r"D:\Cyfronet\unpacked\16 batches\Cutted_pictures"
# path_to_normal_images = r"D:\Cyfronet\unpacked\8 batches\Normal"
send_path = r"D:\cyfronet images\Seperate\4 batches\Normal"
# send_path = r"D:\cyfronet images\Seperate\16 batches\Cutted_pictures"
# send_path = r"D:\cyfronet images\Seperate\8 batches\Normal\Generated"
# img_x_size = 250 # image sizes set by the user
# img_y_size = 150
img_x_size = 250 # image sizes set by the user
img_y_size = 150
# img_x_size = 256 # sdfnasdjbfk;jasdbf;kj
# img_y_size = 256
img_size = (img_x_size, img_y_size)

img_nmbr = 0
for dir in listdir(path_to_normal_images):
    print(path_to_normal_images + "\\" + dir)
    final_path = path_to_normal_images + "\\" + dir + "\\samples\\default"
    gen_img = img_loader(final_path)
    for x in gen_img:
        print(x.shape)
        i = x.shape[1]/img_x_size
        j = x.shape[0]/img_y_size
        images = []
        for l in range(int(j)):
            for k in range(int(i)):
                print("k:", k)
                print("l:", l)
                print(k*img_x_size, img_x_size*(k+1))
                print(x[l*img_y_size:(l+1)*img_y_size, k*img_x_size:img_x_size*(k+1)].shape)
                temp_img = x[l*img_y_size:(l+1)*img_y_size, k*img_x_size:img_x_size*(k+1)].reshape((img_y_size, img_x_size))
                im = Image.fromarray(temp_img).convert('RGB') # needs to be converted to RGB, if not will throw an error
                im.save(send_path +"\\" + dir + "\\\Generated\\" + str(img_nmbr) + str(l) + str(k) + ".bmp")
                print(send_path +"\\" + dir + "\\\Generated\\" + str(img_nmbr) + str(l) + str(k) + ".bmp")
        img_nmbr += 1