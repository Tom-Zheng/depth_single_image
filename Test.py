
# coding: utf-8

# # Depth Map Prediction from Single Image
# 
# Original paper: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
# 

# ## Building Graph

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict_single
import model
import train_operation as op
import csv


COARSE_DIR = "trained_weights/coarse"
REFINE_DIR = "trained_weights/refine"

REFINE_TRAIN = True
FINE_TUNE = True

TEST_DIR = "test_test.csv"
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74


def test():
    # clear old variables
    tf.reset_default_graph()
    global_step = tf.Variable(0, trainable=False)
    # Input one image at a time
    filename = tf.placeholder(tf.string)
    # input
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    
    image = tf.reshape(image,[-1,IMAGE_HEIGHT,IMAGE_WIDTH,3])

    
    # Evaluation Process
    if REFINE_TRAIN:
        coarse_eval = model.inference(image, reuse=False, trainable=False)
        logits_eval = model.inference_refine(image, coarse_eval, 1.0, reuse=False)
    else:
        logits_eval = model.inference(image, reuse=False)
    
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init_op)
        # parameters preloading
        coarse_params = {}
        refine_params = {}
        if REFINE_TRAIN:
            for variable in tf.global_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                elif variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                elif variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        
        # define saver
        
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
        
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")
        
        # Read images
        with open(TEST_DIR) as csvfile:
            reader = csv.DictReader(csvfile)
            index = 0
            for row in reader:
                logits_eval_value, image_value= sess.run([logits_eval, image], feed_dict={filename: row['image']})
                # output training prediction
                output_predict_single(logits_eval_value, image_value, "predict", index)
                index += 1
        
        
def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    test()

if __name__ == '__main__':
    tf.app.run()

