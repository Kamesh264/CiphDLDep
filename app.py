
# # Import necessary libraries
# from flask import Flask, request, render_template
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import pickle
# import pandas as pd
# import numpy as np
# app = Flask(__name__)


# app=Flask(__name__, template_folder='./templates', static_folder='./static')

# model_dl = pickle.load(open('model.pkl','rb'))




# # Define a function to perform image classification
# def classify_image(img_path):
#     img = image.load_img(img_path, target_size=(32, 32))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0  # Normalize the image
#     result = model_dl.predict(img)
#     class_index = np.argmax(result)
#     return class_index

# # Define a route for the main page
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Handle file upload
#         uploaded_file = request.files['file']
#         if uploaded_file.filename != '':
#             img_path = 'uploads/' + uploaded_file.filename
#             uploaded_file.save(img_path)
#             class_index = classify_image(img_path)
#             return f'Predicted class index: {class_index}'
    
#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)

# Import necessary libraries
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


app=Flask(__name__, template_folder='./templates', static_folder='./static')

model_dl = pickle.load(open('model.pkl','rb'))




# Define a function to perform image classification
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32, 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    result = model_dl.predict(img)
    class_index = np.argmax(result)
    return class_index

# Define a route for the main page
@app.route('/index', methods=['GET', 'POST'])
def index():
        
    img1 = request.form.get("imageUplaod");
    class_index = classify_image(img1)
    # return f'Predicted class index: {class_index}'
    return render_template('index.html')

if __name__== '_main_':
    app.run(debug=True)
