from keras.layers import Dense, Lambda, Input
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16

import pdb

from scda_utils import *

class RetrievalModel(object):
    """Unsupervised(use pre-trained model) image retrival model"""
    def __init__(self, model_name='vgg16', input_shape=(16, 224, 224, 3)):
        self.model = self.build_model(input_shape, model_name)

    def build_model(self, input_shape, model_name='vgg16'):
        input_tensor = Input(batch_shape=input_shape) # batch size must be specified

        if model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
            base_model.load_weights('vgg_model.h5')
            map1 = base_model.get_layer('block5_conv2').output
            map2 = base_model.get_layer('block5_pool').output
        else:
            pass

        feat_vec = Lambda(scda_flip_plus)([map1, map2])

        return Model(inputs=base_model.input, outputs=feat_vec)

    def encode(self, images):
        return self.model.predict(images)

class ClassifyModel(object):
    """Image classification model"""
    pass

