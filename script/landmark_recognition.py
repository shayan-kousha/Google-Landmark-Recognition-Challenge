from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
import os, csv, sys, glob, pickle, time
#import cv2 as cv
from PIL import Image

def write_results(ids, landmarks, conf, step, csv_path="../data/submission_axis=1.csv"):
    with open(csv_path, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        if step==0:
            writer.writerow(['id', 'landmarks'])
        
        for im in range(len(ids)):
            writer.writerow([str(ids[im]), str(landmarks[im]) + " " + str(conf[im])])
    
def recognition_model (features, num_classes):
    # conv layer 1
    conv1 = tf.layers.conv2d(inputs=features, filters=48, kernel_size=11, strides=4, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2)
    
    # conv layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=5, strides=1, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2)
    
    # conv layer 3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=192, kernel_size=3, padding='SAME', strides=1, activation=tf.nn.relu)

    # conv layer 4
    conv4 = tf.layers.conv2d(inputs=conv3, filters=192, kernel_size=3, padding='SAME', strides=1, activation=tf.nn.relu)

    # conv layer 5
    conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=3, padding='SAME', strides=1, activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2)

    # conv layer 6
    conv6 = tf.layers.conv2d(inputs=pool5, filters=150, kernel_size=4, strides= 1, activation=None)
    
    _, w, h, k = conv6.shape
    
    logits = tf.reshape(conv6, [-1, w*h*k])
    
    return logits

def accuracy_cal(prediction, y):
    prediction = tf.argmax(prediction, 1)
    correct_pred = tf.equal(prediction, tf.cast(y, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return accuracy
    
def load_patch (images, batch_size, label_dict=None, step=0):
    stack_images = None
    labels = []
    names = []
    
    for im in range(batch_size):
        image_path = images[im + batch_size * step]
        image_name = image_path.split('/')[-1].split('.')[0]
        
        names.append(image_name)
        
        if not(label_dict is None):
            labels.append(int(label_dict[image_name]))
        
        #image = cv.imread(image_path)
        image = Image.open(image_path)
        image = np.array(image)
        
        if stack_images is None:
            stack_images = image[np.newaxis, :, :, :]
        else:
            #print("image shpae = {}".format(image.shape))
            w, h, _ = image.shape
            if w != 512 or h != 512:
                print (image_path)
            stack_images = np.vstack((stack_images, image[np.newaxis, :, :, :]))

    return stack_images, labels, names

def train_loop (label_dict, image_paths, mode):
    # TODO 
    epoch_number = 10
    batch_size = 128
    num_classes = 15000
    
    num_batches = int(np.floor(len(image_paths)/batch_size))
    
    f = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    l = tf.placeholder(tf.int32, [batch_size])
    
    pr = tf.placeholder(tf.float32, [batch_size, num_classes])
    label = tf.placeholder(tf.int32, [batch_size])
    
    #model = recognition_model(f, l, num_classes)
    model = recognition_model(f, num_classes)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=l, logits=model)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0000005)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    accuracy = accuracy_cal(pr, label)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/model_epoch_5.ckpt")

        if int(mode) == 0:
            print("{} Start training...".format(datetime.now()))
            start_time = time.time()
            
            for epoch in range(6, epoch_number):
                sum_loss = 0
                count = 0
                
                for step in range(num_batches):
                    images, labels, names = load_patch(image_paths, batch_size, label_dict, step)
                    logits, _, ret_loss = sess.run([model, train_op, loss], feed_dict={f: images, l: labels})
                    acc = sess.run(accuracy, feed_dict={pr: logits, label: labels})
                    
                    sum_loss += ret_loss
                    count += 1
                    
                    if step % 100 == 0:
                        print("Epoch {}, Step {}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, step, sum_loss/count, acc))
                        count = sum_loss = 0
                    
                    
                print("epoch = {}, execution time = {}".format(epoch, str(time.time() - start_time)))
                print(len(tf.get_default_graph().get_operations()))
                
                save_path = saver.save(sess, "./checkpoints/model_epoch_" + str(epoch) + ".ckpt" )
        
        elif int(mode) == 1:
            print("{} Start predicting...".format(datetime.now()))
            
            for step in range(num_batches):
                images, _, names = load_patch(image_paths, batch_size, step=step)
                logits = sess.run(model, feed_dict={f: images})
                prediction = tf.argmax(logits, 1)
                pred_conf = tf.reduce_max(tf.nn.softmax(logits, axis=1), 1)
                write_results(ids=names, landmarks=prediction.eval(), conf=pred_conf.eval(), step=step)
                
                if step % 100 == 0:
                    print("Step {}".format(step))
                
def main (argv):
    if len(argv) != 3:
        print('Syntax: %s <image_path> <mode/>' % sys.argv[0])
        sys.exit(0)
        
    (image_path, mode) = argv[1:]

    image_paths = glob.glob(image_path + "*.jpg")

    with open('../data/train.pkl', 'rb') as handle:
        label_dict = pickle.load(handle)
    
    train_loop(label_dict, image_paths, mode)

if __name__ == "__main__":
    tf.app.run()
    