from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50




basemodel = ResNet50(include_top=False, pooling='avg')
a = basemodel.output
b = Dense(1)(a)
model = Model(inputs=basemodel.input, outputs=b)
model.layers[0].trainable = False
print(model.summary())


model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(batch_size=32, x=train_X, y=train_Y, epochs=30)
