from flask import Flask, render_template, request, send_file, Response
import os
import requests
import logging
from model import U2NET # full size version 173.6 MB
from PIL import Image
import io
import torch
from torch.autograd import Variable
from torchvision import transforms #, utils
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT
from data_loader import ToTensor
import cv2
import time

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)
app.logger.debug('this will show in the log')

model_name='u2net'#u2netp
model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'
# --------- 2. dataloader ---------
#eval_transforms = transforms.Compose([RescaleT(320), ToTensor()])
# --------- 3. model define ---------
if(model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
elif (model_name == 'u2netp'):
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3, 1)
model_load_start_time = time.time()
net.load_state_dict(torch.load(model_dir, map_location='cpu'))
#net.load_state_dict(torch.load(model_dir))
app.logger.debug("Model Load Time: %s seconds ---" % (time.time() - model_load_start_time))
net.eval()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def remove_background(input_image):

    # --------- 1. get image path and name ---------
    #model_name='u2net'#u2netp
    #model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'
    Output_Image_Path = './test_data/u2net_results/Alpha_Blending/out1.png'


    # --------- 2. dataloader ---------
    # eval_transforms = transforms.Compose([RescaleT(320), ToTensor()])
    # # --------- 3. model define ---------
    # if (model_name == 'u2net'):
    #     print("...load U2NET---173.6 MB")
    #     net = U2NET(3, 1)
    # elif (model_name == 'u2netp'):
    #     print("...load U2NEP---4.7 MB")
    #     net = U2NETP(3, 1)
    # model_load_start_time = time.time()
    # net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # # net.load_state_dict(torch.load(model_dir))
    # print("Model Load Time: %s seconds ---" % (time.time() - model_load_start_time))
    #
    # net.eval()
    eval_transforms = transforms.Compose([RescaleT(320), ToTensor()])
    # --------- 4. inference for each image ---------
    #The below code takes the uploaded image and creates an alpha mask

    #Convert PIllow Image to OpenCV
    opencv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGRA)

    #Put opencv_image into dict so u2net tensor functions will work
    input_image_dict = {'imidx': np.array([[0]]), 'image': opencv_image, 'label': opencv_image}

    #Transform images so it is same size as model was trained on
    image_tensor = eval_transforms(input_image_dict)
    input = Variable(image_tensor['image'])
    d1, d2, d3, d4, d5, d6, d7 = net(input[None, ...].float())

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    im_np = np.array(im)
    w, h = input_image.size
    imo = im.resize((w, h), resample=Image.BILINEAR)
    pb_np = np.array(imo)
    del d1, d2, d3, d4, d5, d6, d7

    #Below code is to take the alpha mask and remove the background from image
    #https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
    foreground = opencv_image
    #background will be image that you want to combine the foreground with.
    background = np.zeros([foreground.shape[0], foreground.shape[1], 3], dtype=np.uint8)
    background = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGRA)
    print(foreground.shape)
    print(background.shape)
    background.fill(255)
    alpha = pb_np

    #This will take the foreground and mask and create a png image of the foreground with transparent background.
    b, g, r = cv2.split(alpha)
    print(b.shape)
    foreground[:, :, 3] = b
    #cv2.imshow("Output_Image", foreground)
    #cv2.waitKey(0)
    #cv2.imwrite(Output_Image_Path, foreground)





    # Convert uint8 to float
    #foreground = foreground.astype(float)
    #background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    #alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    #foreground = cv2.multiply(alpha, foreground)


    # Multiply the background with ( 1 - alpha )
    #background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    #outImage = cv2.add(foreground, background)

    # Display image
    #cv2.imwrite(Output_Image_Path, outImage)
    #cv2.imshow("Output_Image",cv2.resize(outImage/255, (1500,1000)))
    #cv2.imshow("Output_Image", outImage/255 )
    #cv2.waitKey(0)

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGRA2RGBA)
    foreground_pil = Image.fromarray(foreground)

    return foreground_pil




@app.route('/')
def index():
    return "PicSpotlight API"


@app.route('/remove_background_api', methods=['POST'])
def remove_background_api():
    print("Remove Background function called.................")
    if request.method == 'POST':
        print("Post received")
        img = Image.open(request.files['file'].stream)
        #img.save('./static/uploads/upload.jpg')
        img_bg_removed = remove_background(img)
        #img_bg_removed.save('./test_data/API Images/out.png')
        # img.save('./static/uploads/upload.jpg')
        img_bg_io = io.BytesIO()
        img_bg_removed.save(img_bg_io, 'PNG', quality=100)
        img_bg_io.seek(0)
        resp = Response(img_bg_io, status=200)
        #r = requests.post('http://127.0.0.1:5000/receive_image', files={'file': img_bg_io.getvalue()})
        #print(r)
        img.close()
        return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0')
