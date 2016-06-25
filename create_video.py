import cv2
import os
import numpy as np
from images2gif import writeGif
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
    base_dim = 100
    width = num_width*base_dim
    height = num_height*base_dim

    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png')], key=lambda x: img_id(x))
    print(file_names)
    new_im = Image.new('RGB', (width,height))
    # print(len(file_names))
    index = 0
    for i in xrange(0,height,base_dim):
        for j in xrange(0,width,base_dim):
            im = Image.open(os.path.join(images_root,file_names[index]))
            im.thumbnail((base_dim,base_dim))

            draw = ImageDraw.Draw(im)
            # font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")
            draw.text( (50,60), str(index+1), fill="#000000")#, font=font)
            del draw
            new_im.paste(im, (j,i))
            index += 1

    new_im.save(os.path.join(images_root,os.path.basename(images_root)+".png"))


def create_grid_mutation(images_root, num_width=20, num_height=1):
    # overwrite the method
    def img_id(filename):
        begin = len('changing_')
        end = filename.find('_amount')
        return int(filename[begin:end])

    def change_amount(filename):
        begin = filename.find('amount_')+len('amount_')
        end = filename.find('.png')
        return float(filename[begin:end])

    base_dim = 40
    width = num_width*base_dim
    height = num_height*base_dim
    border = 4

    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png') and 'batch' not in fn], key=lambda x: img_id(x))#[::4]
    print(file_names)
    new_im = Image.new('RGB', (width+(num_width+1)*border,height+(num_height+1)*border))
    # print(len(file_names))
    new_height = height+base_dim+border if num_width == 1 else height
    new_width = width+base_dim+border if num_height == 1 else width

    index = 0
    for i in xrange(0,new_height,base_dim+border):
        for j in xrange(0,new_width,base_dim+border):
            print(i,j)
            im = Image.open(os.path.join(images_root,file_names[index]))
            im.thumbnail((base_dim,base_dim))

            draw = ImageDraw.Draw(im)
            # font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")

            label = index+1
            label = change_amount(file_names[index])


            draw.text( (5,30), str(label), fill="#FFFFFF")#, font=font)
            del draw
            im = ImageOps.expand(im,border=border,fill='red')
            new_im.paste(im, (j,i))
            index += 1

    new_im.save(os.path.join(images_root,os.path.basename(images_root)+".png"))

if __name__ == '__main__':
    # root = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation'
    # images_root = 'ballsregL2_subsample_3_dim_hidden_64_heads_2_sharpening_rate_10_L2_0.001_learning_rate_0.0009_feature_maps_16_numballs_2'
    # for exp in [f for f in os.listdir(os.path.join(root,images_root)) if '.txt' not in f and not f.startswith('.')]:
    #     for demo in [f for f in os.listdir(os.path.join(*[root,images_root,exp])) if not f.startswith('.')]:
    #         print demo
    #         create_gif(os.path.join(*[root, images_root, exp, demo]))

    # create_gif('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/18_layers_3_lrdecay_0.99_dataset_folder_14_235balls_lr_0.0003_sharpen_1_model_lstmobj/videos')
    # create_gif('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/17_layers_3_sharpen_1_lr_0.0003_lrdecay_0.99/videos')
    # create_grid('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/18_layers_3_lrdecay_0.99_dataset_folder_14_235balls_lr_0.0003_sharpen_1_model_lstmobj/videos')

    # create_grid_mutation('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation/ballsvar2heads2_Apr_01_15_57/ballsvar2_heads_2_learning_rate_0.0003_feature_maps_16_subsample_3_dim_hidden_64_L2_0_numballs_2/batch_100_input_11_along_17')
    # create_grid_mutation('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation/ballsvar2heads2_Apr_01_15_57/ballsvar2_heads_2_learning_rate_0.0003_feature_maps_16_subsample_3_dim_hidden_64_L2_0_numballs_2/batch_75_input_11_along_1')
    # create_grid_mutation('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation/ballsvar2heads2_Apr_01_15_57/ballsvar2_heads_2_learning_rate_0.0003_feature_maps_16_subsample_3_dim_hidden_64_L2_0_numballs_2/batch_75_input_6_along_52')


    # root = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation/balls2headsreg/ballsregL2_subsample_3_dim_hidden_64_heads_2_sharpening_rate_10_L2_0.001_learning_rate_0.0009_feature_maps_16_numballs_2/'
    # create_grid_mutation(root + 'batch_40_input_15_along_30')
    # create_grid_mutation(root + 'batch_30_input_1_along_18')
    # create_grid_mutation(root + 'batch_80_input_1_along_16')
    # create_grid_mutation(root + 'batch_60_input_1_along_16')


    create_gif('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/18_layers_3_lrdecay_0.99_dataset_folder_14_34balls_lr_0.0003_sharpen_1_model_lstmobj/videos')
