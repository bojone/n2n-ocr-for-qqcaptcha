import glob
import numpy as np
from scipy import misc
from keras.layers import Input, Convolution1D, MaxPooling1D, Flatten, Activation, Dense
from keras.models import Model
from keras.utils.np_utils import to_categorical


imgs = glob.glob('sample/*.jpg')
img_size = (53, 129)
data = np.array([misc.imresize(misc.imread(i, flatten=True), img_size).T for i in imgs])
data = 1 - data.astype(float)/255.0
target = np.array([[ord(i)-ord('a') for i in j[7:11]] for j in imgs])
target = [to_categorical(target[:,i], 26) for i in range(4)]
img_size = img_size[::-1]

input = Input(img_size)
cnn = Convolution1D(64, 3)(input)
cnn = Convolution1D(64, 3)(cnn)
cnn = MaxPooling1D(2)(cnn)
cnn = Activation('relu')(cnn)
cnn = Convolution1D(32, 2)(cnn)
cnn = MaxPooling1D(2)(cnn)
cnn = Convolution1D(32, 2)(cnn)
cnn = MaxPooling1D(2)(cnn)
cnn = Flatten()(cnn)
cnn = Activation('relu')(cnn)

model = Model(input=input, output=[Dense(26, activation='softmax')(Dense(52, activation='relu')(cnn)) for i in range(4)])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 256
nb_epoch = 300
model.fit(data, target, batch_size=batch_size, nb_epoch=nb_epoch)
model.save_weights('yanzheng_cnn_2d.model')
rr = [''.join(chr(i.argmax()+ord('a')) for i in model.predict(data[[k]])) for k in tqdm(range(len(data)))]
s = [imgs[i][7:11]==rr[i] for i in range(len(imgs))]
print 1.0*sum(s)/len(s)
