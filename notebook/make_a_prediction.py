from keras.models import load_model
import tensorflow as tf
import PIL.Image as img
import numpy as np
from tensorflow.keras.preprocessing import image


# CHANGE the model_path
model_path = "C:\\Users\\wegas\\Documents\\6. Kuliah Semester 6\\Bangkit Belajar\\Repository\\model_transfer_learning.h5-20230608T025206Z-001\\model_transfer_learning.h5"
modelx = tf.saved_model.load(model_path)


# Importing image yang akan di prediksi
# CHANGE the image path
img = image.load_img('../dataset_predict/'+'MONAS3.png', target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = x /255.

# Prediction
prediction = modelx(x)


# Output prediksi
class_labels = ['ampera','gadang','gwk','kotatua','monas','ulundanu']
label = class_labels[np.argmax(prediction)]
print(f"Prediksi: {label} \ndengan tingkat Probabilitas: {np.max(prediction)}")