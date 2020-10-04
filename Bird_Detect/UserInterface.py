
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

#!/usr/bin/env python
import datetime
from importlib import import_module
import os
import flask
from flask import Flask, render_template, Response, request, redirect, make_response, session



def bird():

    window = tk.Tk()

    window.title("Bird Detection")

    window.geometry("520x420")
    window.configure(background ="lightblue")

    title = tk.Label(text="     Click below to choose picture of Bird", background = "lightblue", fg="Brown", font=("Lucida Grande",  15))

    title.grid()



    def analysis():
        # USAGE
        # python classify.py --model pokedex.model --labelbin lb.pickle --image examples/00000012.jpg

        # import the necessary packages
        from keras.preprocessing.image import img_to_array
        from keras.models import load_model
        import numpy as np
        import argparse
        import imutils
        import pickle
        import cv2
        import os

        # construct the argument parse and parse the arguments
        '''ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
        ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
        ap.add_argument("-i", "--image", required=True,
                help="path to input image")
        args = vars(ap.parse_args())'''

        # load the image

        path="img\\1.jpg"
        image = cv2.imread(path)
        output = image.copy()
         
        # pre-process the image for classification
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network and the label
        # binarizer
        print("[INFO] loading network...")
        model = load_model("pokedex.model")
        lb = pickle.loads(open("lb.pickle", "rb").read())

        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        # we'll mark our prediction as "correct" of the input image filename
        # contains the predicted label text (obviously this makes the
        # assumption that you have named your testing image files this way)
        filename = path[path.rfind(os.path.sep) + 1:]
        correct = "correct" if filename.rfind(label) != -1 else ""

        # build the label and draw the label on the image
        label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # show the output image
        print("[INFO] {}".format(label))
        cv2.imshow("Output", output)
        cv2.waitKey(0)






    def openphoto():
        dirPath = "img"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
        fileName = askopenfilename(initialdir='birdpics', title='Select image for analysis ',filetypes=[('image files', '.jpg')])
        dst = "img\\1.jpg"
        shutil.copy(fileName, dst)
        load = Image.open(fileName)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render, height="250", width="500")
        img.image = render
        img.place(x=0, y=0)
        img.grid(column=0, row=1, padx=10, pady = 10)
        title.destroy()
        button1.destroy()
        button2 = tk.Button(text="Analyse Image",bg='#0052cc', fg='#ffffff',command = analysis)
        button2.grid(column=0, row=2, padx=10, pady = 10)



    button1 = tk.Button(text="Get Photo", command = openphoto,height=5,width=25,bg='#0052cc', fg='#ffffff',font=('algerian',10,'bold'))


    button1.grid(column=0, row=500, padx=40, pady = 10)



    window.mainloop()


app = Flask(__name__)
@app.errorhandler(404)
def fun1(e):
    return flask.redirect("/")
@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == "POST":
        bird()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)



"""
def bird():
	window = tk.Tk()

	window.title("Bird Detection")

	window.geometry("520x420")
	window.configure(background ="lightblue")

	title = tk.Label(text="     Click below to choose picture of Bird", background = "lightblue", fg="Brown", font=("Lucida Grande",  15))
	title.grid()


	button1 = tk.Button(text="Get Photo", command = openphoto,height=5,width=25,bg='#0052cc', fg='#ffffff',font=('algerian',10,'bold'))


	button1.grid(column=0, row=500, padx=40, pady = 10)



	window.mainloop()
 
bird()

"""






