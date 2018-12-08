"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor，仅得到该张量的引用，并非计算该张量
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#              
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                embeddings, label_batch, paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, args.distance_metric, args.subtract_mean,
                args.use_flipped_images, args.use_fixed_image_standardization)

              
def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels，入队一批图片和标签
    nrof_embeddings = len(actual_issame)*2          # nrof_pairs * nrof_images_per_pair，全部原始图片的数量=pairs数量×一个paris图片数量
    nrof_flips = 2 if use_flipped_images else 1     #是否采用翻转图片
    nrof_images = nrof_embeddings * nrof_flips      #输入到模型的图片数量，包括翻转图片（一张原图片对应一个翻转图片）

    labels_array = np.expand_dims(np.arange(0,nrof_images),1)       #[0,1,...,5999],变为列向量
    # 图片路径列向量，repeat是为了放置翻转图片path，注意，repeat的行为和concatenate的行为不同，前者重复元素挨着，后者是成块拼接，这个决定了模型输出向量的拼接方式
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)           #控制每个图片预处理的向量，和labels_array同形，列向量

    if use_fixed_image_standardization:                             #采用均值127.5的标准化处理过程到(0,1)，而不是tf自带的x-mean/std的标准化
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP            #labels_array % 2 :[0,1,0,1,...]，每间隔2张图片，有一个需要翻转

    #输入队列（图片path，图片标签，图片预处理方式）
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])                 #特征向量维度
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size                        #总批数 = 样本数//每批样本数
    emb_array = np.zeros((nrof_images, embedding_size))             #全部样本的嵌入向量
    lab_array = np.zeros((nrof_images,))                            #全部样本对应的label
    #分成batch进行推断，mini_batch
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:                                             #每10个batch就显示进度.
            print('.', end='')
            sys.stdout.flush()
    print('')

    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))     #原始图片和翻转图片的嵌入向量拼接成最终特征向量
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]           #注意：np.repeate(image_paths,2)方式，导致原图片和翻转图片是相邻的[a,a,b,b],不是[a,b...a,b]
        embeddings[:,embedding_size:] = emb_array[1::2,:]           #embeddings的宽度是emb_array宽度的2倍，emb_array[1::2]，从1开始每间隔2取样
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate(TAR): %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)                     #from sklearn import metrics，给定tpr,fpt计算auc，绘制roc
    print('Area Under Curve (AUC): %1.3f' % auc)
    # brentq是scipy.optimize中混合求根方法，先用一种方法求解，如果速度不快，则寻找另外更好的方法，加速求解
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)   #等效错误率，从(0,1)到(1,0)的直线和ROC曲线的交点，其横坐标就是EER
    print('Equal Error Rate (EER): %1.3f' % eer)            #EER也就是FPR=FNR的值，由于FNR=1-TPR，可以画一条从（0,1）到（1,0）的直线，找到交点

    import matplotlib.pyplot as plt
    plt.plot(fpr,tpr,'-')
    plt.title('ROC')
    plt.show()

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
