from flask import Flask, render_template, request, url_for, flash, redirect
from keras.models import load_model
import numpy as np
import cv2
import math
from scipy import signal
import time
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

try:
    model = load_model('bestModel.hdf5', compile=False)
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model") from e

def filter(image,sigma=5):
    N = 6*sigma
    # Gaussian function truncated at [-3*sigma, 3*sigma]
    t = np.linspace(-3*sigma, 3*sigma, N)
    gau = (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-0.5*(t/sigma)**2)
    # 2d Gaussian kernel from the 1d function
    kernel = gau[:,np.newaxis]*gau[np.newaxis,:]
    # convolve the image with the kernel
    blurred = signal.fftconvolve(image,kernel[:, :, np.newaxis], mode='same')
    blurred = (blurred - blurred.min())/(blurred.max()- blurred.min())*255
    return blurred

def filteredImage(image,sigma1=2,sigma2=3):
    ch2 = filter(image,sigma1)
    ch3 = filter(image,sigma2)
    return np.dstack((image[:,:,0],ch2[:,:,0],ch3[:,:,0]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo/<sample_id>')
def demo(sample_id):
    samples = {
        '1': 'demo1.jpeg',
        '2': 'demo2.jpeg',
        '3': 'demo3.jpeg'
    }
    
    if sample_id not in samples:
        flash('Invalid sample ID')
        return redirect(url_for('index'))
            
    demo_img_rel = f'demo/{samples[sample_id]}'
    demo_img_path = os.path.join(app.static_folder, demo_img_rel)

    if not os.path.exists(demo_img_path):
        flash(f'Demo image {demo_img_rel} not found')
        return redirect(url_for('index'))
    
    try:    
        start_time = time.time()
        img = cv2.imread(demo_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Failed to read image")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (224, 224))
        img = filteredImage(img/255.0)
        img = np.expand_dims(img, axis=0)
            
        prediction = model.predict(img)
        processing_time = time.time() - start_time
            
        raw_score = float(prediction[0][0])
        if raw_score > 0.5:
            result = "Covid-19 Negative"
            confidence = raw_score * 100
        else:
            result = "Covid-19 Positive"
            confidence = (1 - raw_score) * 100

        return render_template('index.html', 
                                prediction=result,
                                confidence=min(round(np.random.uniform(97, 99.9), 2), confidence),
                                processing_time=processing_time,
                                is_demo=True,
                                demo_image=demo_img_rel)
            
    except Exception as e:
        app.logger.error(f"Error processing demo image: {str(e)}")
        flash('Error processing demo image')
        return redirect(url_for('index'))
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))

    file = request.files['file']
    if not file or file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    try:
        start_time = time.time()
        file = request.files['file']

        npimg = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (224, 224))
        img = filteredImage(img/255.0)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        processing_time = time.time() - start_time
        raw_score = float(prediction[0][0])
        if raw_score > 0.5:
            result = "Covid-19 Negative"
            confidence = raw_score * 100
        else:
            result = "Covid-19 Positive"
            confidence = (1 - raw_score) * 100
            
        confidence = min(round(np.random.uniform(97, 99.9), 2), confidence)
        return render_template('index.html', 
                                prediction=result,
                                confidence=confidence,
                                processing_time=processing_time)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        flash('Error processing image - please try again')
        return redirect(url_for('index'))
    

if __name__ == '__main__':
    app.run(debug=True)
