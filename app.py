from flask import Flask, render_template, request, send_file, Response
import os
import requests
import logging
from u2net_test import remove_background

from PIL import Image
import io

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)
app.logger.debug('this will show in the log')

@app.route('/')
def index():
    return "PicSpotlight API"

@app.route('/test', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        img = Image.open(request.files['file'].stream)
        #img.save('./static/uploads/upload.jpg')
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=100)
        img_io.seek(0)
        r = requests.post('http://127.0.0.1:5000/process_image', files={'file': img_io.getvalue()})
        print(r.text)

        return "OK"



@app.route('/remove_background_api', methods=['POST'])
def remove_background_api():
    if request.method == 'POST':
        print("Post received")
        img = Image.open(request.files['file'].stream)
        #img.save('./static/uploads/upload.jpg')
        img_bg_removed = remove_background(img)
        img_bg_removed.save('./test_data/API Images/out.png')
        # img.save('./static/uploads/upload.jpg')
        img_bg_io = io.BytesIO()
        img_bg_removed.save(img_bg_io, 'PNG', quality=100)
        img_bg_io.seek(0)
        resp = Response(img_bg_io, status=200)
        #r = requests.post('http://127.0.0.1:5000/receive_image', files={'file': img_bg_io.getvalue()})
        #print(r)
        return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0')
