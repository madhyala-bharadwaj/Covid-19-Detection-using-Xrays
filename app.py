from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import math
from scipy import signal

app = Flask(__name__)
model = load_model('bestModel.hdf5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    s = "<h2>This trained Covid-19 Detection Model works with an accuracy of 99% for training data and 100% accuracy for Validation Data</h2>"
    return s

@app.route('/predict', methods=['POST'])
def predict():
    def filter(image,sigma=5):
        N = 6*sigma
        # create a Gaussian function truncated at [-3*sigma, 3*sigma]
        t = np.linspace(-3*sigma, 3*sigma, N)
        gau = (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-0.5*(t/sigma)**2)
        # create a 2d Gaussian kernel from the 1d function
        kernel = gau[:,np.newaxis]*gau[np.newaxis,:]
        # convolve the image with the kernel
        blurred = signal.fftconvolve(image,kernel[:, :, np.newaxis], mode='same')
        blurred = (blurred - blurred.min())/(blurred.max()- blurred.min())*255
        return blurred
    def filteredImage(image,sigma1=2,sigma2=3):
        ch2 = filter(image,sigma1)
        ch3 = filter(image,sigma2)
        return np.dstack((image[:,:,0],ch2[:,:,0],ch3[:,:,0]))
    if request.method == 'POST':
        file = request.files['file']
        if file:
            npimg = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (224, 224))
            img = filteredImage(img/255.0)
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            result = "Covid-19 Negative" if prediction[0][0] > 0.5 else "Covid-19 Positive"
            return render_template('index.html',prediction=result)
        return render_template('index.html', prediction="Failed to read file ! Try again")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)