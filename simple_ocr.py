#
# You can modify this files
#
import tensorflow as tf
import random
from models import YoloV3Tiny
from dataset import transform_images, load_tfrecord_dataset
from absl import app, flags, logging
import numpy as np
class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = YoloV3Tiny(classes=3)
        self.model.load_weights('./checkpoints/yolov3_train_15.tf')
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']
        self.class_names = [c.strip() for c in open('names.names').readlines()]

    # TODO: implement find label
    def find_label(self, img):

        img = tf.expand_dims(img, 0)
        img = transform_images(img, 416)
        boxes, scores, classes, nums = self.model(img)
        for i in range(nums[0]):
        #     logging.info('\t{}, {}, {}'.format(self.class_names[int(classes[0][i])],
        #                                        np.array(scores[0][i]),
        #                                        np.array(boxes[0][i])))
        # print("classes: ", self.class_names[int(classes[0][i])])
            print(i)
        #     if self.class_names[int(classes[0][i])] == 'starbuck':
        #         return "starbucks"
        #     else :
        #         return "others"
        # if classes == '':
        #     classes = "others"
