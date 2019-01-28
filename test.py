import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import PIL
import os

import pdb

from data_loader import DataLoader
from models import RetrievalModel

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

DATAPATH = "d:/data_all/front100" 

def encode(data, model):
    """encode the images in support data into vectors before retrieval"""
    support_vecs = model.encode(data)
    np.save('supvecs.npy', support_vecs)

def retrieve(image, model, num=5):
    """retrieve the most similar image"""
    image = np.expand_dims(image, axis=0) 
    target_vec = model.encode(image) 
    support_vecs = np.load('supvecs.npy')

    distances = np.sum(np.square((support_vecs - target_vec)), axis=-1)
    indices = distances.argsort(axis=0)[:num]
    return indices

def evaluate(loader, model):
    encode(loader.trainx, model)

    top1 = 0
    top5 = 0
    for i in range(loader.testx.shape[0]):
        inds = retrieve(loader.testx[i], model, num=5)
        labels = loader.trainy[inds]
        if loader.testy[i] == labels[0]: 
            top1 += 1
        if loader.testy[i] in labels: # not categorical
            top5 += 1
    top1 /= loader.testx.shape[0] 
    top5 /= loader.testx.shape[0] 
    
    print("top1 acc:%s, top5 acc: %s"%(top1, top5))



if __name__ == "__main__":
    loader = DataLoader(datapath=DATAPATH) 
    model = RetrievalModel() 

    evaluate(loader, model)

    