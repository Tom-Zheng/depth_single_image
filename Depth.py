
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
from dataset import output_predict
import model
import train_operation as op

MAX_STEPS = 100
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 12
TRAIN_FILE = "train.csv"
EVAL_FILE = "eval.csv"

# TODO: Adapt to current job
COARSE_DIR_READ = "coarse"
REFINE_DIR_READ = "refine"

COARSE_DIR = "coarse"
REFINE_DIR = "refine"

TRAINING_SET_SIZE = 1080
EVAL_SET_SIZE = 120

EPOCH_NUM = 500

REFINE_TRAIN = True
FINE_TUNE = True

def train():
    # clear old variables
    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)

    # Training Process
    dataset = DataSet(BATCH_SIZE)
    images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)
    if REFINE_TRAIN:
        print("refine train.")
        coarse = model.inference(images, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv)
    else:
        print("coarse train.")
        logits = model.inference(images)
    loss = model.loss(logits, depths, invalid_depths)

    # Evaluation Process
    dataset_eval = DataSet(BATCH_SIZE)
    images_eval, depths_eval, invalid_depths_eval = dataset.csv_inputs(EVAL_FILE)

    if REFINE_TRAIN:
        coarse_eval = model.inference(images_eval, reuse=True, trainable=False)
        logits_eval = model.inference_refine(images_eval, coarse_eval, 1.0, reuse=True)
    else:
        logits_eval = model.inference(images_eval, reuse=True)
        
    loss_eval = model.loss(logits_eval, depths_eval, invalid_depths_eval)

    train_op = op.train(loss, global_step, BATCH_SIZE)
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
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR_READ)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR_READ)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")
        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in xrange(EPOCH_NUM):
            index = 0
            for i in xrange(TRAINING_SET_SIZE/BATCH_SIZE):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                # Print loss
                if index % 10 == 0:
                    loss_eval_value = 0.0
                    for j in xrange(EVAL_SET_SIZE/BATCH_SIZE):
                        loss_eval_value += sess.run(loss_eval, feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
                    loss_eval_value = loss_eval_value/(EVAL_SET_SIZE/BATCH_SIZE)
                    print("%s: %d[epoch]: %d[iteration]: train loss %f; eval loss %f" % (datetime.now(), step, index, loss_value, loss_eval_value))
                    print('{{"metric": "loss", "value": {}}}'.format(loss_value))
                    print('{{"metric": "eval_loss", "value": {}}}'.format(loss_eval_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    with open('/output/log.csv', 'a') as output:
                        output.write("%d,%d,%f,%f" % (step,index,loss_value,loss_eval_value))
                        output.write("\n")
                index += 1
            # output training prediction
            if REFINE_TRAIN:
                output_predict(logits_val, images_val, "/output/training_output/predict_refine_%05d" % step)
            else:
                output_predict(logits_val, images_val, "/output/training_output/predict_%05d" % step)
            
            if (step+1) % 5 == 0 or (step + 1) == EPOCH_NUM:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        
        
def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()

if __name__ == '__main__':
    tf.app.run()

