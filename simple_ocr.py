#
# You can modify this files
#
import tensorflow as tf
import random
from models import YoloV3Tiny
from dataset import transform_images, load_tfrecord_dataset
from absl import app, flags, logging
import numpy as np
import cv2
from utils import draw_outputs
class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = YoloV3Tiny(classes=3)
        # self.model.load_weights('./checkpoints/yolov3_train_15.tf')
        self.model.load_weights('./checkpoints-test/yolov3_train_19.tf')

        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']
        self.class_names = [c.strip() for c in open('names.names').readlines()]

    # TODO: implement find label
    def find_label(self, img1, count):
        img = tf.expand_dims(img1, 0)
        img = transform_images(img, 416)
        boxes, scores, classes, nums = self.model(img)
        img = cv2.cvtColor(img1.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        cv2.imwrite('output/' + str(count) + '.jpg', img)
        if nums:
            for i in range(nums[0]):
                print(np.array(scores[0][i]))
                label = self.class_names[int(classes[0][i])]
                if label != "highlands" and np.array(scores[0][i]) < 0.01:
                    return "others"
                else:
                    return label
        else:
            return "others"
