import os 
import math
import glob 
from PIL import Image
import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex,to_rgb 
from config import CONFIG 
import cv2
import numpy as np 
import regex as re

matplotlib.use('Agg')
data  = 'masa'

if data  == 'dubai':
    labels = ['#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929', '#9B9B9B','#000000' ]
    labels = [int(to_rgb(l)[0]*255) for l in labels]
    cls = range(1,len(labels)+1)
    labels = dict(zip(labels,cls))
    print("Splitting dataset into smaller images") 

    base_path = CONFIG.MISC_dataset_path
    source = os.path.join(base_path,'dubai_ful')
    destination = os.path.join(base_path,'dubai_out')
    if os.path.exists(destination): 
        print('Destination file already exists')
    else: 
        os.mkdir(destination) 
        os.mkdir(os.path.join(destination,'images'))
        os.mkdir(os.path.join(destination,'labels'))

    images_path = glob.glob(os.path.join(source,'*/images/*'))
    masks_path = [i[:-25] +'masks'+i[-19:-3] + 'png' for i in images_path]

    i = 1 
    for image_path,mask_path in zip(images_path,masks_path):
        
        image = plt.imread(image_path)
        mask = plt.imread(mask_path)
    #    mask = (mask[:,:,0]*255).astype(int)
    #    for i_h in range(mask.shape[0]):
    #        for i_w in range(mask.shape[1]):
    #            mask[i_h,i_w] = labels[mask[i_h,i_w]]

        size = 500
        n_H = math.floor(image.shape[0]/size)
        n_W = math.floor(image.shape[1]/size)
        
        #image = cv2.resize(image,(n_H*256,n_W*256))
        #mask = cv2.resize(mask,(n_H*256,n_W*256))
        for i_h in range(n_H):
            for i_w in range(n_W): 
                im  = image[i_h*size:(i_h+1)*size,i_w*size:(i_w+1)*size,:]
                m  = mask[i_h*size:(i_h+1)*size,i_w*size:(i_w+1)*size,:]
                plt.imsave(destination +  f"/images/{i:04d}.jpg",im)
                plt.imsave(destination +  f"/labels/{i:04d}.jpg",m)
                
                print(f"{i:04d}")
                i += 1     

elif data  == 'masa': 
    print("Splitting dataset into smaller images") 

    base_path = CONFIG.MISC_dataset_path
    source = os.path.join(base_path,'masa_full')
    destination = os.path.join(base_path,'masa_seven')
    if os.path.exists(destination): 
        print('Destination file already exists')
    else: 
        os.mkdir(destination) 
        os.mkdir(os.path.join(destination,'image'))
        os.mkdir(os.path.join(destination,'label'))

    images_path = glob.glob(os.path.join(source,'*tiff/test/*')) + glob.glob(os.path.join(source,'*tiff/train/*')) + glob.glob(os.path.join(source,'*tiff/val/*'))
    pattern  = '(train|val|test)'
    masks_path = [re.sub(pattern, r'\g<1>_labels', i)[:-1] for i in images_path]
    zro = []
    i = 1 
    for image_path,mask_path in zip(images_path,masks_path):
        
        image = plt.imread(image_path)
        mask = plt.imread(mask_path)
    #    mask = (mask[:,:,0]*255).astype(int)
    #    for i_h in range(mask.shape[0]):
    #        for i_w in range(mask.shape[1]):
    #            mask[i_h,i_w] = labels[mask[i_h,i_w]]
        dim = 700
        n_H = math.floor(image.shape[0]/dim)
        n_W = math.floor(image.shape[1]/dim)
        
        #image = cv2.resize(image,(n_H*256,n_W*256))
        #mask = cv2.resize(mask,(n_H*256,n_W*256))
        for i_h in range(n_H):
            for i_w in range(n_W): 
                im  = image[i_h*dim:(i_h+1)*dim,i_w*dim:(i_w+1)*dim,:]
                m  = mask[i_h*dim:(i_h+1)*dim,i_w*dim:(i_w+1)*dim,:]
                zro_tmp = (im.sum(axis=-1) == 765).sum()
                if zro_tmp < 100:
                    plt.imsave(destination +  f"/image/{i:04d}.jpg",im)
                    plt.imsave(destination +  f"/label/{i:04d}.jpg",m)
                    print(f"{i:04d}")
                    i += 1     

#                if  zro_tmp < 2000:
#                    zro += [zro_tmp]
#                    if zro_tmp >100  :
#                        plt.imshow(im)
#                        print(zro_tmp)
#                        plt.show()
#                
#    plt.hist(zro, bins=50)
#    plt.show()
