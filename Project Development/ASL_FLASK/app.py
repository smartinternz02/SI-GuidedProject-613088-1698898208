
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template

app = Flask(__name__)

model = load_model("ASL_DenseNet.h5", compile = False)
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath, target_size = (108, 108)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis = 0)
        print(x)
        y = model.predict(x)
        preds = np.argmax(y, axis=1)
        print("prediction",preds)
        index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        text = "The predicted letter is : " + str(index[preds[0]])
    return text

if __name__ == '__main__':
    app.run(debug = False, threaded = False)

