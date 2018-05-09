import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import h5py 
from keras.models import model_from_json  


#读取model  
model=model_from_json(open('D:\\code\\NotePad\\my_model_architecture.json').read())  
model.load_weights('D:\\code\\NotePad\\my_model_weights.h5') 


target_size = (229, 229) #fixed size for InceptionV3 architecture
imgName = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\Images\\AF1.jpg"
img_rows = 350
img_cols = 350
# img = Image.open(imgName)
# modelPath = "D:\\code\\NotePad\\ResNet50.model"
# model = load_model(modelPath)


img = Image.open(imgName)
# im = img.resize(target_size)
im = np.array(img)
im = np.expand_dims(im, axis=0)
# im = im.reshape(1,img_rows, img_cols,3)
preds = model.predict(im)


