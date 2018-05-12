from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
import os, csv, sys, glob, pickle, time
import cv2 as cv

def write_results(ids, landmarks, step, csv_path="../data/submission.csv"):
    with open(csv_path, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        if step==0:
            writer.writerow(['id', 'landmarks'])
        
        for im in range(len(ids)):
            writer.writerow([str(ids[im]), str(landmarks[im])])
    
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

    b, w, h, k = pool5.shape
    
    # dense layer 1
    flattened = tf.reshape(pool5, [-1, w*h*k])
    dense1 = tf.layers.dense(inputs=flattened, activation=tf.nn.relu, units=w*h*k)
    # TODO dropout
    
    # dense layer 2
    dense2 = tf.layers.dense(inputs=dense1, activation=tf.nn.relu, units=w*h*k)
    # TODO dropout
    
    # dense layer 3
    logits = tf.layers.dense(inputs=dense2, units=num_classes)
    
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
        
        image = cv.imread(image_path)
        
        if stack_images is None:
            stack_images = image[np.newaxis, :, :]
        else:
            stack_images = np.vstack((stack_images, image[np.newaxis, :, :]))

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

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    accuracy = accuracy_cal(pr, label)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/model.ckpt")

        if int(mode) == 0:
            print("{} Start training...".format(datetime.now()))
            start_time = time.time()
            
            for epoch in range(epoch_number):
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
                #print(names)
                #print(prediction.eval())
                write_results(ids=names, landmarks=prediction.eval(), step=step)
                
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
    