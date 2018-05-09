import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dense, Activation,GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras import optimizers

import h5py
ImgPath2 = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\Images\\"
img_rows = 350
img_cols = 350

NB_ResNet_LAYERS_TO_FREEZE = 49
def process_line(line,ImgPath):

	index = line.index('g')  
	imgName = line[0:index+1]
	label = line[index+2:index+7]
	img = ImgPath+imgName
	
	im = np.array(Image.open(img))
	label_np = np.array(label)
	
	im = im.reshape(1,img_rows, img_cols,3)
	label_np = label_np.reshape(1)
	
	return im,label_np


def generate_arrays_from_file(textPath,ImgPath):

    while 1:
        f = open(textPath,"r")
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line,ImgPath2)
			# print(x)
            yield (x, y)
        f.close()

	# return x
def load_data(textPath,ImgPath):
	x_img = []
	y_label = []
	
	file = open(textPath,"r") 
	# data = file.readlines()
	for line in file:
		index = line.index('g')  
		imgName = line[0:index+1]
		label = line[index+2:index+7]
		img = ImgPath+imgName
		im = np.array(Image.open(img))
		x_img.append(im) 
		label_np = np.array(label)
		y_label.append(label_np)
	# print(im)	
	return x_img,y_label

def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_ResNet_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_ResNet_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])	

def train(textPath,ImgPath):
	basemodel = ResNet50(include_top=False, pooling='avg')
	a = basemodel.output
	#a = GlobalAveragePooling4D()(a)
	b = Dense(1)(a)
	model = Model(inputs=basemodel.input, outputs=b)
	#setup_to_finetune(model)
	model.layers[0].trainable = False
	print(model.summary())

	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=adam)
	
	history_ft = model.fit_generator(generate_arrays_from_file(textPath,ImgPath),
                    steps_per_epoch=1000, epochs=10)
	# model.save("D:\\code\\NotePad\\ResNet50.model")
		#保存神经网络的结构与训练好的参数
	json_string = model.to_json()#等价于 json_string = model.get_config()  
	open('D:\\code\\NotePad\\my_model_architecture.json','w').write(json_string)    
	model.save_weights('D:\\code\\NotePad\\my_model_weights.h5')

if __name__ == "__main__":    

	textPath = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\train_test_files\\split_of_60%training and 40%testing\\train.txt"
	textPath_test = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\train_test_files\\split_of_60%training and 40%testing\\test.txt"
	ImgPath = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\Images\\"

	train(textPath,ImgPath)
	
