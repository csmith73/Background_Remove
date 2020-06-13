import os
import cv2
import time
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def blend(foreground, background, alpha):
    """This function composes a new image for given foreground image, background image and alpha matte.

    This is done by applying the composition equation

    .. math::
        I = \\alpha F + (1-\\alpha)B.

    Parameters
    ----------
    foreground: numpy.ndarray
        Foreground image
    background: numpy.ndarray
        Background image
    alpha: numpy.ndarray
        Alpha matte

    Returns
    -------
    image: numpy.ndarray
        Composed image as numpy.ndarray

    Example
    -------
    >>> from pymatting import *
    >>> foreground = load_image("data/lemur/lemur_foreground.png", "RGB")
    >>> background = load_image("data/lemur/beach.png", "RGB")
    >>> alpha = load_image("data/lemur/lemur_alpha.png", "GRAY")
    >>> I = blend(foreground, background, alpha)
    """
    if len(alpha.shape) == 2:
        print("alpha len(alpha.shape) == 2")
        alpha = alpha[:, :, np.newaxis]
    print(alpha.shape)
    print(foreground.shape)
    print(background.shape)


    return alpha * foreground + (1 - alpha) * background



def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp


    image_dir = './test_data/test_images/'
    prediction_dir = './test_data/' + model_name + '_results/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    img_name_list = glob.glob(image_dir + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        #save_output(img_name_list[i_test],pred,prediction_dir)
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        #im = Image.fromarray(predict_np * 255).convert('RGB')
        im = Image.fromarray(predict_np * 255).convert('RGB')
        im_np = np.array(im)
        print("im_np shape")
        print(im_np.shape)
        img_name = img_name_list[i_test].split("/")[-1]
        image = io.imread(img_name_list[i_test])
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        pb_np = np.array(imo)
        print("pb_np_shape")
        print(pb_np.shape)


        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        #imo.save(d_dir + imidx + '.png')

        del d1,d2,d3,d4,d5,d6,d7
        print(type(imo))


        #img = cv2.imread(img_name_list[i_test])
        #img = np.array(img)
        #background = Image.new('RGB', (img.shape[1], img.shape[0]), (255, 255, 255))
        #background = np.array(background)

        #cv2.imshow("Origninal Image",img)
        #cv2.waitKey(0)
        #pb_np = pb_np[:, :, 2]
        #cv2.imshow("Alpha Mat Image", pb_np/255)
        #cv2.waitKey(0)
        #print(pb_np)
        #pb_np = pb_np/255
        #new_image = blend(img, background, pb_np)
        #cv2.imshow("New Image", new_image)
        #cv2.waitKey(0)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #mask = cv2.imread(pred)
        #mask = cv2.cvtColor(pb_np, cv2.COLOR_BGR2GRAY)
        #transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        #transparent[:, :, 0:3] = img
        #transparent[:, :, 3] = mask
        #img_write_path = "./test_data/test_images/" + str(img_name_single)
        #print(img_write_path)
        #print(img_name_list[i_test])
        file_name_orig = img_name_list[i_test].split("/")[-1]
        file_name_orig = file_name_orig.split("\\")[1]
        file_name_orig = file_name_orig.split(".")[0]
        file_name = file_name_orig + "_Mask.png"
        print(file_name)
        #cv2.imwrite("./test_data/u2net_results/" + file_name, new_image)
        #imo.save("./test_data/u2net_results/Image_Masks/" + file_name)
        cv2.imwrite("./test_data/u2net_results/Image_Masks/" + file_name, pb_np)
        time.sleep(2)
        # Read the images
        Original_Image_Path = './test_data/test_images/' + file_name_orig + '.jpg'
        print(Original_Image_Path)
        Mask_Image_Path = './test_data/u2net_results/Image_Masks/' + file_name
        Output_Image_Path = './test_data/u2net_results/Alpha_Blending/' + file_name_orig + '.png'
        foreground = cv2.imread(Original_Image_Path)
        #foreground = pb_np

        background = np.zeros([foreground.shape[0], foreground.shape[1], 3], dtype=np.uint8)
        background.fill(255)
        # background = np.array(background)
        # background = cv2.imread("./test_data/test_data/llama.jpg")
        alpha = cv2.imread(Mask_Image_Path)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        # Display image
        # cv2.imshow("outImg", outImage / 255)
        # cv2.waitKey(0)
        cv2.imwrite(Output_Image_Path, outImage)

if __name__ == "__main__":
    main()
