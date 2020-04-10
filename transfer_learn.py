"""Perform transfer-learning for offset stratification with a provided COVID-Net

From the trained weights of a COVID-Net for COVID-19 identification in radiographs,
this tool performs transfer learning to re-use these weights for stratification of
patient offset (# of days since symptoms began *)

Steps to use this:
1. follow instructions for building data dir with train & test subdirs
2. train your network with the train_tf.py script
3. run this script and pass

(*) FIXME: It seems that the definition of offset varies between data sources! (account for this)
TODO: Make this script more general so that it can be used to transfer learn for other applications

paul@darwinai.ca
"""
import argparse
import cv2
import os
import pathlib
from typing import List
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf

from eval import eval
from data import BalanceDataGenerator


# We will create a checkpoint which has initial values for these variables
VARS_TO_FORGET = [
    'dense_3/kernel:0',
    'dense_3/bias:0',
    'dense_2/kernel:0',
    'dense_2/bias:0',
    'dense_1/kernel:0',
    'dense_1/bias:0',
]
IMAGE_SHAPE = (224, 224, 3)
DISPLAY_STEP = 1


def get_parse_fn(num_classes: int):
    def parse_function(imagepath: str, label: int):
        """Parse a single element of the stratification dataset"""
        image_decoded = tf.image.resize_images(
            tf.image.decode_jpeg(tf.io.read_file(imagepath), IMAGE_SHAPE[-1]), IMAGE_SHAPE[:2])
        return (
            tf.image.convert_image_dtype(image_decoded, dtype=tf.float32) / 255.0, # x
            tf.one_hot(label, num_classes), # y
            tf.convert_to_tensor(1.0, dtype=tf.float32), # sample_weights TODO: verify this is right
        )
    return parse_function


if __name__ == "__main__":

    # Input args
    parser = argparse.ArgumentParser(description='COVID-Net Transfer Learning Script (offset).')
    parser.add_argument('--classes', default=4, type=int,
                        help='Number of classes to stratify offset into.')
    parser.add_argument('--stratification', type=int, nargs='+', default=[3, 5, 10],
                        help='Stratification points (days), i.e. "5 10" produces stratification of'
                        ': 0o <-0c-> 5o <-1c-> 10o -2c-> via >= comparison (o=offset, c=class).')
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number of epochs (less since we\'re effectively fine-tuning).')
    parser.add_argument('--lr', default=0.00002, type=float, help='Learning rate.')
    parser.add_argument('--bs', default=8, type=int, help='Batch size')
    parser.add_argument('--inputweightsdir', default='models/COVIDNetv2', type=str,
                        help='Path to input folder containing a trained COVID-Net checkpoint')
    parser.add_argument('--inputmetafile', default='model.meta', type=str,
                        help='Name of meta file within <inputweightsdir>')
    parser.add_argument('--outputdir', default='models/CovidRiskNet', type=str,
                        help='Path to output folder.')
    parser.add_argument('--trainfile', default='train_COVIDx.txt', type=str,
                        help='Name of train file.')
    parser.add_argument('--testfile', default='test_COVIDx.txt', type=str,
                        help='Name of test file.')
    parser.add_argument('--name', default='COVIDRiskNet', type=str,
                        help='Name of folder to store training checkpoints.')
    parser.add_argument('--chestxraydir', default='../covid-chestxray-dataset', type=str,
                        help='Path to the chestxray images directory for COVID-19 patients.')
    args = parser.parse_args()

    # Check inputs
    assert os.path.exists(args.inputweightsdir), "Missing file {}".format(args.inputweightsdir)
    assert os.path.exists(os.path.join(args.inputweightsdir, args.inputmetafile)), \
        "Missing file {}".format(args.inputmetafile)

    # Format and define a stratification method based on our points
    assert len(args.stratification) > 1, "Must pass more than one offset stratification point"
    if args.stratification[0] != 0:
        stratification = np.array([0, *args.stratification])
    else:
        stratification = np.array(args.stratification)
    num_classes = len(stratification)
    stratify = lambda offset: np.where(offset >= stratification)[0][-1]

    # Read CSV of dataset
    assert os.path.exists(args.chestxraydir), "please clone "\
        "https://github.com/ieee8023/covid-chestxray-dataset and pass path to dir as --chestxraydir"
    csv = pd.read_csv(os.path.join(args.chestxraydir, "metadata.csv"), nrows=None)

    # We need to read the offsets for COVID patients in our split
    # FIXME: ideally we should just store the offset in the split as well or read it from CSV by id.
    # FIXME: we need a different split or more data - existing split does not respect
    train_files, train_labels, test_files, test_labels = [], [], [], []
    for is_training, split_txt in zip([True, False], [args.trainfile, args.testfile]):
        for split_entry in open(split_txt).readlines():
            _, image_file, diagnosis = split_entry.strip().split()
            if diagnosis == 'COVID-19':
                patient = csv[csv["filename"] == image_file]
                recorded_offset = patient['offset'].item()
                if not np.isnan(recorded_offset):
                    offset = stratify(int(recorded_offset))
                    image_path = os.path.abspath(
                        os.path.join(args.chestxraydir, 'images', image_file))
                    assert os.path.exists(image_path), "Missing file {}".format(image_path)
                    if is_training:
                        train_files.append(image_path)
                        train_labels.append(offset)
                    else:
                        test_files.append(image_path)
                        test_labels.append(offset)

    assert len(train_files) >= 0 and len(train_labels) >= 0, "Got no train cases"
    assert len(test_files) >= 0 and len(test_labels) >= 0, "Got no test cases"
    print("collected {} training and {} test cases for transfer-learning".format(
        len(train_files), len(test_files)))

    # Init augmentation fn - FIXME: we need a way to put this in a parse_fn for tf.data.dataset
    # augmentation_fn = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    #     brightness_range=(0.9, 1.1),
    #     fill_mode='constant',
    #     cval=0.,
    # )
    # < define generator from augmentation_fn + cv loads? >
    # dataset = tf.data.Dataset.from_generator(lambda: generator,
    #                                       output_types=(tf.float32, tf.float32, tf.float32),
    #                                       output_shapes=([batch_size, 224, 224, 3],
    #                                                      [batch_size, 3],
    #                                                      [batch_size]))
    # TODO: we need a training method that we can re-use. below very similar to train_tf.py

    # Output path creation for this run with lr param in name
    train_dir = os.path.join(args.outputdir, args.name + '-lr' + str(args.lr))
    os.makedirs(args.outputdir, exist_ok=True)
    os.makedirs(train_dir)
    print('Output: ' + train_dir)

    # Train
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        # Import meta graph
        tf.train.import_meta_graph(os.path.join(args.inputweightsdir, args.inputmetafile))

        # Restore pre-trained vars which are not in our VARS_TO_FORGET list
        restore_vars_list, init_vars_list = [], []
        for var in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if var.name in VARS_TO_FORGET:
                init_vars_list.append(var)
            else:
                restore_vars_list.append(var)
        restore_saver = tf.train.Saver(var_list=restore_vars_list)
        restore_saver.restore(sess, tf.train.latest_checkpoint(args.inputweightsdir))

        # Get I/O tensors
        image_tensor = graph.get_tensor_by_name("input_1:0")
        labels_tensor = graph.get_tensor_by_name("dense_3_target:0")
        sample_weights = graph.get_tensor_by_name("dense_3_sample_weights:0")
        pred_tensor = graph.get_tensor_by_name("dense_3/MatMul:0")

        # Define tf.dataset
        existing_vars = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        dataset = dataset.map(get_parse_fn(num_classes))
        dataset = dataset.shuffle(15).batch(batch_size=args.bs).repeat()
        iterator = dataset.make_initializable_iterator()
        gn_op = iterator.get_next()

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred_tensor, labels=labels_tensor)*sample_weights)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op)
        optim_vars = list(
            set(sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(existing_vars))

        # Initialize the optimizer + dsi + vars in our VARS_TO_FORGET list
        sess.run(tf.variables_initializer(optim_vars + init_vars_list))

        # save base model
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(train_dir, 'model'))
        print('Saved baseline checkpoint')
        #print('Baseline eval:')
        #eval(sess, graph, test_files, 'test')  # FIXME: need eval for this network

        # Training cycle
        print('Transfer Learning Started.')
        print('\t{} train samples\n\t{} test samples\n'.format(len(train_files), len(test_files)))
        sess.run(iterator.initializer.name)
        num_batches = len(train_files) // args.bs
        progbar = tf.keras.utils.Progbar(num_batches)
        for epoch in range(args.epochs):
            for i in range(num_batches):

                # Run optimization
                batch_x, batch_y, weights = sess.run(gn_op)
                sess.run(train_op, feed_dict={image_tensor: batch_x,
                                              labels_tensor: batch_y,
                                              sample_weights: weights})
                progbar.update(i+1)

            if epoch % DISPLAY_STEP == 0:
                pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                loss = sess.run(
                    loss_op,
                    feed_dict={
                        pred_tensor: pred,
                        labels_tensor: batch_y,
                        sample_weights: weights,
                    }
               )
                print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
                # eval(sess, graph, test_files, 'test')
                saver.save(
                    sess,
                    os.path.join(train_dir, 'model'),
                    global_step=epoch+1,
                    write_meta_graph=False
                )
                print('Saving checkpoint at epoch {}'.format(epoch + 1))

    print("Transfer Learning Finished!")
