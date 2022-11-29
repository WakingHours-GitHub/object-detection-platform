import os
from flask import Flask, render_template, request, Response
import cv2 as cv
import numpy as np
from PIL import Image
# from ssd.ssd import SSD
# from ssd import *
from yolov5.yolo import YOLO
from yolov5 import *

import base64
from utils.camera import *

import sys 
from pathlib import Path
runtime_path = sys.path[0]
print(runtime_path)

app = Flask(__name__, static_folder=os.path.join(runtime_path, "./static"))


file_name = ['jpg','jpeg','png']
video_name = ['mp4','avi']

yolo = YOLO()
yolo
@app.route('/images', methods=['POST'])
def main_route():  # put application's code here
    try:
        for ele in os.listdir(os.path.join(runtime_path, "file")):
            ele_path = os.path.join(runtime_path, "file", ele)
            os.remove(ele_path)
        pass
    except:
        print("file not delect")
    # fatch image
    image = request.files["images"]
    image_name = image.filename
    file_path = os.path.join(runtime_path, "file", image_name)
    image.save(file_path)

    print("file_download_path", file_path)
    if image_name.split(".")[-1] in file_name: # 如果是图片的化
        source_img = cv.imread(file_path)
        h, w, _ = source_img.shape
        # if h > 2000 or w > 2000:
            # h = h // 2
            # w = w // 2
            # source_img = cv.resize(source_img, (int(w), int(h)))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(source_img)) # 转换为Image. 
        img = np.array(yolo.detect_image(img))
        hstack_img = np.hstack([source_img, img])
        cv.imwrite(file_path, hstack_img)
        

        _, img_encoded = cv.imencode('.jpg', hstack_img)
        response = img_encoded.tobytes()


        try:
            img_stream = ''
            # img_stream = cv.imencode('.jpg', img)[1].tobytes()
            with open(file_path, 'rb') as img_f:
                img_stream = img_f.read()
                img_stream = base64.b64encode(img_stream).decode()
            # return Response(response=response, status=200, mimetype='image/jpg')
            return render_template('image_process.html', image_url=img_stream)
        except:
            return render_template('image_process.html')

    elif image_name.split(".")[-1] in video_name: # 如果是视频:
        print("video upload")
        with open(os.path.join(runtime_path, "./file_name.txt"), "w") as f_writer:
            f_writer.write(image_name)
            # f_writer.close()
        
        return render_template("video_process.html")


    return render_template("index.html")


@app.route('/')
def main_page():
    return render_template("index.html")



@app.route('/realtime')
def realtime():  # put application's code here
    return render_template("realtime.html")


@app.route('/about_me')
def about_me():  # put application's code here
    return render_template("about_me.html")

@app.route('/whole_frame')
def whole_frame():
    file_path = os.path.join(os.getcwd(), "templates/Construction.jpg")
    
    img_stream = ''
    with open(file_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return render_template("whole_frame.html", img_stream=img_stream)





def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b''b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(Camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
