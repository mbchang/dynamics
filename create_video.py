import cv2
import os
import numpy as np
from images2gif import writeGif
from PIL import Image, ImageDraw, ImageFont
import os
import pprint

# Usage: Download images2gif
# Change
#   for im in images:
#       palettes.append( getheader(im)[1] )
# To
#   for im in images:
#       palettes.append(im.palette.getdata()[1])

def img_id(filename):
    return int(filename[-7:-4])

def create_gif(images_root):
    """
        writeGif(filename, images, duration=0.1, loops=0, dither=1)
            Write an animated gif from the specified images.
            images should be a list of numpy arrays of PIL images.
            Numpy images of type float should have pixels between 0 and 1.
            Numpy images of other types are expected to have values between 0 and 255.
    """
    # def img_id(filename):
    #     begin = len('changing_')
    #     end = filename.find('_amount')
    #     return int(filename[begin:end])


    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png')], key=lambda x: img_id(x))
    images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    filename = os.path.join(images_root, "gif.GIF")
    # print filename
    writeGif(filename, images, duration=0.2)


def create_grid(images_root):

    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png')], key=lambda x: img_id(x))

    new_im = Image.new('RGB', (2000,400))
    # print(len(file_names))
    index = 0
    for i in xrange(0,400,100):
        for j in xrange(0,2000,100):
            im = Image.open(os.path.join(images_root,file_names[index]))
            im.thumbnail((100,100))

            draw = ImageDraw.Draw(im)
            # font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")
            draw.text( (50,60), str(index+1), fill="#000000")#, font=font)
            del draw
            new_im.paste(im, (j,i))
            index += 1


    new_im.save(os.path.join(images_root,os.path.basename(images_root)+".png"))

if __name__ == '__main__':
    # root = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation'
    # images_root = 'ballsgss3_Feb_23_08_10'
    # for exp in [f for f in os.listdir(os.path.join(root,images_root)) if '.txt' not in f and not f.startswith('.')]:
    #     for demo in [f for f in os.listdir(os.path.join(*[root,images_root,exp])) if not f.startswith('.')]:
    #         print demo
    #         create_gif(os.path.join(*[root, images_root, exp, demo]))

    # create_gif('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/18_layers_3_lrdecay_0.99_dataset_folder_14_235balls_lr_0.0003_sharpen_1_model_lstmobj/videos')
    # create_gif('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/17_layers_3_sharpen_1_lr_0.0003_lrdecay_0.99/videos')
    create_grid('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/18_layers_3_lrdecay_0.99_dataset_folder_14_235balls_lr_0.0003_sharpen_1_model_lstmobj/videos')
