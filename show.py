import cv2
from keras.layers import Lambda, Input, Dense
from keras.models import Model, load_model
from keras import optimizers
import pdb
from keras import backend as K
import numpy as np
from keras.utils import plot_model 

from scda import select_aggregate
from keras.applications.vgg16 import VGG16 

def show(): 
    test = cv2.imread('test.jpg') 
    test = cv2.resize(test, (224, 224))
    test = test.reshape((1, 224, 224, 3))

    cnn = VGG16(weights='imagenet', include_top=False, input_shape=[224, 224, 3], pooling='avg')
    cnn.load_weights('vgg_model.h5')

    masklayer = Lambda(lambda x: select_aggregate(x, [224, 224]))(cnn.get_layer('block5_pool').output)
    model = Model(inputs=cnn.input, outputs=masklayer)

    output = model.predict(test)
    cropped = ((output * test)[0]).astype('uint8')
    
    cv2.imshow('img', cropped)
    cv2.imwrite("cropped.jpg", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show()