# This file will use to score your implementations.
# You should not change this file

import os
import pandas as pd
import time
import sys
import cv2
import tensorflow as tf

from simple_ocr import HoadonOCR
if __name__ == "__main__":

    input_folder = sys.argv[1]
    label_file = sys.argv[2]

    df_labels = pd.read_csv(label_file)
    img_name = df_labels['img_name'].values.tolist()
    img_labels = df_labels['label'].values.tolist()
    img_to_label = dict([(img_name[i], img_labels[i]) for i in range(len(img_name))])

    start_time = time.time()
    model = HoadonOCR()
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))
    fail_process = 0
    cnt_predict = 0

    start_time = time.time()
    count = 0
    for filename in list_files:
        count = count + 1
        img = cv2.imread(os.path.join(input_folder, filename))
        # print(img.shape)
        height, width, channels = img.shape
        img_crop = img[0:int(height/2), 0:width]
        cv2.imwrite('sampledata-crop/' + str(count) + '.jpg', img_crop)
        img = tf.image.decode_image(
            # open(os.path.join(input_folder, filename), 'rb').read(), channels=3)
            open(os.path.join('sampledata-crop/', str(count) + '.jpg'), 'rb').read(), channels = 3)

        try:
            label = model.find_label(img, count)
        except:
            label = -1
        print("Label detected: ", label)
        print("True label:", img_to_label[filename])
        if img_to_label[filename] == label:
            cnt_predict += 1
        elif label == -1:
            fail_process += 1

    run_time = time.time() - start_time

    print("Ket qua dung: %i/%i" % (cnt_predict, len(list_files)))
    print("Loi: %i" % fail_process)
    print("Score = %.2f" % (10.*cnt_predict/len(list_files)))
    print("Run time in: %.2f s" % run_time)
