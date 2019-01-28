import os
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import pdb

def color_process(data):
    """color preprocessing"""
    return data / 128. - 1.
        
def image_process(img):
    """transform an image to an array, and add a batch axis"""
    data = image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    return data

class DataLoader(object):
    """The data loader class
    datapath: the path to load images
    savepath: the path to save npy files
    npypath: the path to load npy files
    """
    def __init__(self, input_size=(224, 224, 3), datapath=None, savepath=None, npypath=None):
        if datapath != None:
            # load images
            self.input_size = input_size
            self.datapath = datapath 
            self.classes = os.listdir(datapath)
            self.le = LabelEncoder()
            self.le.fit(self.classes)

            print("[INFO] data path is %s"%self.datapath)
            print("[INFO] classes are: %s"%" ".join(self.classes))
            print("[INFO] loading data...")
            self.trainx, self.trainy, self.testx, self.testy = self.load_data()
            print("[INFO] data loaded!")
            print("[INFO] data statistics: train: %s test: %s"%(self.trainx.shape[0], self.testx.shape[0]))


        if npypath != None:
            # load npy file
            #self.loadnpy()
            pass

        if savepath != None:
            # save data as npy file
            #self.savenpy
            pass

    def load_folder(self, path, label, padding=True):
        """load images from data path with label information."""
        imglist = os.listdir(path)
        for imagefile in imglist: 
            print("loading image file %s"%('/'.join([path, imagefile])))
            img = Image.open(os.path.join(path, imagefile))
            img = np.asarray(img.convert("RGB"))

            if padding == True: #padding to keep the h/w ratio
                h, w = img.shape[0], img.shape[1]
                max_size = max(h, w)
                img = cv2.copyMakeBorder(img, (max_size-h)//2, (max_size-h)//2, (max_size-w)//2, (max_size-w)//2, cv2.BORDER_CONSTANT, value = [0,0,0]) 
            img = cv2.resize(img, (self.input_size[0], self.input_size[1]))

            datax = image_process(img)
            if imglist.index(imagefile)==0:
                DataX = datax
            else:
                DataX = np.vstack((DataX,datax))
        DataY = [label] * len(imglist)
        DataY = self.le.transform(DataY)
        DataY = DataY.reshape(DataY.shape[0], 1)
        #DataY = to_categorical(DataY, num_classes=len(self.classes))
        return DataX, DataY
    
    def load_data(self, ratio=0.8):     
        """load train and test data. depends on specific task
        ratio: ratio of train/all
        """
        DataX, DataY = self.load_folder(os.path.join(self.datapath, self.classes[0]), self.classes[0]) 
        for class_name in self.classes[1:]:
            datax, datay = self.load_folder(os.path.join(self.datapath, class_name), class_name)
            DataX = np.vstack((DataX, datax))
            DataY = np.vstack((DataY, datay)) 
        
        indices = [i for i in range(DataX.shape[0])]
        np.random.shuffle(indices) # use sklearn?

        DataX = DataX[indices]
        DataY = DataY[indices]

        TrainX = DataX[:int(len(indices) * ratio)] 
        TrainY = DataY[:int(len(indices) * ratio)] 
        TestX = DataX[int(len(indices) * ratio):] 
        TestY = DataY[int(len(indices) * ratio):] 

        TrainX = color_process(TrainX)
        TestX = color_process(TestX)
                
        return TrainX, TrainY, TestX, TestY

    def loadnpy(self):
        pass 
    
    def savenpy(self):
        pass 

#if __name__ == "__main__":
#    loader = DataLoader()