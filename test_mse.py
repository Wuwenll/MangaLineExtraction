import argparse
import os
import numpy as np

import fnmatch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2

from keras.models import model_from_json


from keras import backend as K


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch, 0])
    return activations


def loadImages(folder):
    imgs = []
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
    return matches

class dataLoader():
    def __init__(self,folder, batch_size = 1):
        self.root = folder
        matches = []
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, '*'):
                matches.append(os.path.join(root, filename))
        self.images = matches
        self.batch_size = batch_size
        self.now_point = 0

    def have_next_batch(self):
        if self.now_point == len(self.images):
            return False
        else:
            return True

    def next_batch(self):
        imgs = []
        next_point = min(self.now_point+self.batch_size,len(self.images))
        for i in range(self.now_point,next_point):
            src = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
            patch = np.empty((1, 1, src.shape[0], src.shape[1]), dtype="float32")
            patch[0, 0, :, :] = np.ones((src.shape[0], src.shape[1]), dtype="float32") * 255.0
            patch[0, 0, 0:src.shape[0], 0:src.shape[1]] = src
            imgs.append(patch)
        return_img = np.concatenate(imgs,axis=0)
        self.now_point = next_point
        return  return_img

batch_size = 1


def loadModel():
    # load json and create model
    json_file = open('./erika_tf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./erika_unstable_tf.h5")
    return model



def split_image(image, num=5):
   h, w = image.shape
   result = []
   for i in range(num):
      result.append(image[:, i * w // num:w // num * (i + 1)])
   return result

        rows = int(src.shape[0]/16 + 1)*16
        cols = int(src.shape[1]/16 + 1)*16

    

        patch = np.empty((1,1,rows,cols),dtype="float32")
        patch[0,0,:,:] = np.ones((rows,cols),dtype="float32")*255.0
        patch[0,0,0:src.shape[0],0:src.shape[1]] = src

        out = model.predict(patch, batch_size=batch_size)
        if isinstance(out, list):
            out = out[0]

            result_img = out[0,:,:,0]
            input_img = patch[0,:,:,0]

            result_img[result_img > 255] = 255
            result_img[result_img < 0] = 0
            cv2.imwrite(os.path.join(os.path.join(output, book,pg[:-4]+".png")),np.hstack([result_img]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,  help='')
    parser.add_argument('--output_path', type=str, help='')
    args = parser.parse_args()
    test(args)