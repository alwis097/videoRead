from flask import Flask, render_template, Response
# from camera import Video
import cv2


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        faces=faceDetect.detectMultiScale(frame, 1.3, 5)
        for x,y,w,h in faces:
            x1,y1=x+w, y+h
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
            cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
            cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

            cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
            cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

            cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
            cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

            cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
            cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
# from flask import Flask
# from flask import Flask, flash, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd




# UPLOAD_FOLDER = 'static/uploads/'

# app = Flask(__name__)
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# import os




# @app.route('/',)
# def my_form_homepage_web():
#     return render_template('web.html')

# @app.route('/upload')
# def upload():
#     return render_template('upload.html')

# @app.route('/index')
# def index():
#     return render_template('index.html')

# def gen(camera):
#     while True:
#         frame=camera.get_frame()
#         yield(b'--frame\r\n'
#        b'Content-Type:  image/jpeg\r\n\r\n' + frame +
#          b'\r\n\r\n')

# @app.route('/video')

# def video():
#     return Response(gen(Video()),
#     mimetype='multipart/x-mixed-replace; boundary=frame')





from cgitb import html
from distutils.command.upload import upload
from fileinput import filename
import os
from re import X
# from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/',)
def my_form_homepage_web():
    return render_template('web.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_form():
	return render_template('upload.html')
def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')
@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
@app.route('/', methods=['POST'])
def upload_video():

	try:
		text = request.form['text']
		print(text)
		def get_frames(filename):
			print("video")
			print(filename)
			print("video")
			video=cv2.VideoCapture(filename)
			print(video)

			while video.isOpened():
				print("frame")
				rete,frame=video.read()
				print(frame)
				if rete:
					print("video")
					yield frame
				else:
					print("video1")
					break
				video.release()
				yield None

		def get_frame(filename,index):
			counter=0
			# print(filename)
			# print(index)
			video=cv2.VideoCapture(filename)
			while video.isOpened():
				rete,frame=video.read()
				if rete:
					if counter==index:
						return frame
					counter +=1
				else:
					print("break")
					break
			video.release()
			return None


		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)
		else:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#print('upload_video filename: ' + filename)
			flash('Video successfully uploaded and displayed below')


			# Video_FILE = '/uploads/' + filename
			Video_FILE = "static/uploads/"+filename
			print(Video_FILE)
			for f in get_frames(Video_FILE):
				print(f)
				if f is None:
					print("empty")
					break		
				cv2.imshow('frame',f)
				print("empty1")
				if cv2.waitKey(10) == 40:
					print("empty2")
					break
			cv2.destroyAllWindows()


			text1=int(text)
			frame = get_frame(Video_FILE,text1)
			frame1 = "All frames in video ",frame
			# print(frame)
			x = "Shape for frame ", text1," shape is ",frame.shape

			def display(frame):
				return plt.imshow(frame)
			y=display(frame)
			print(y)
			plt.imshow(frame)
			bb =  "Pixels at particular frames"
			# import base64
			# from io import BytesIO

			# tmpfile2 = BytesIO()
			# plt.savefig(tmpfile2, format='png',bbox_inches='tight')
			# plt.savefig('y' , bbox_inches='tight')
			# plt.show()

			# encodedf = base64.b64encode(tmpfile2.getvalue()).decode('utf-8')
			# # print(encodedf)
			# html = '<img src=\'data:image/png;base64,{}\'>'.format(encodedf)
			
	# C:\Users\DELL\OneDrive\Desktop\Tutorial 7\y.png
			# html = '<img src=\'data:image/png;base64,{}\'>'.format(encodedf)
			# html = '<img src=C:\Users\DELL\OneDrive\Desktop\Tutorial 7\y.png;base64,{}\>'.format(encodedf)
			video=cv2.VideoCapture(Video_FILE)
			count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			video.release()
			z='Total no of frames: ',count
			thislist = {}
			try:
				for a in range(1,200):
					for b in range(1,200):
						c='pixel at ',(a,b),frame[a,b,:]
					thislist[(a,b)] = c

			except:
				d="limit exceed"
			new1 = pd.DataFrame.from_dict(thislist)
			new2 = new1.T
			x1 = new2.to_html(escape=False)
			return render_template('upload.html', filename=filename,x=x,frame1=frame1,html=html,z=z,thislist=x1,bb=bb)
	except:
		bb="Insert a frame number or Video size is high"
		return render_template('upload.html',bb=bb)


@app.route('/display/<filename>')
def display_video(filename):

	# print('display_video filename: ' + Video_FILE)
	
	return redirect(url_for('static',filename='uploads/' + filename), code=301)


# if __name__ == "__main__":
#     app.run()
app.run(debug=True)
