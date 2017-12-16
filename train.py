#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml
import math
import sys


tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("decay_coefficient", 100, "Decay coefficient (default: 2.5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

dataset_name = cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
datasets = None
if dataset_name == "mrpolarity":
    datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                    cfg["datasets"][dataset_name]["negative_data_file"]["path"])
elif dataset_name == "20newsgroup":
    datasets = data_helpers.get_datasets_20newsgroup(subset="train",
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
elif dataset_name == "localdata":
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
x_words, x_tags, x_labels, x_trees, x_indices, y, _ = data_helpers.load_data_labels('/u/a/n/anant/Dropbox/539_project/data/')

# Build vocabularies
max_document_length = max([len(x.split(" ")) for x in x_words])
max_document_length = 50
valid_indices = []
for i in range(len(x_words)):
    if len(x_words[i].split(" ")) <= max_document_length:
        valid_indices.append(i)
        
print(max_document_length)
words_vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_words = np.array(list(words_vocab_processor.fit_transform(x_words)))
for i in range(max(max_document_length, len(x_words))):
    if x_indices[i] < max_document_length:
        x_words[i][int(x_indices[i])] = 0
tags_vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_tags = np.array(list(tags_vocab_processor.fit_transform(x_tags)))
labels_vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_labels = np.array(list(labels_vocab_processor.fit_transform(x_labels)))
x_indices = np.array(x_indices)
x_trees = np.array(x_trees)
# x_trees = x_trees.reshape(len(x_words), -1)
x_feats = (list(zip(x_words, x_tags, x_labels, x_indices, x_trees)))

x_feats = np.array([x_feats[i] for i in valid_indices])
y = np.array([y[i] for i in valid_indices])
# vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
# sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
# vocabulary = list(list(zip(*sorted_vocab))[0])
# print("Vocabulary : ")
# print(vocabulary)
# print(x[0])
# print(x)
# print(y)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x_feats[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(words_vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train[0][0].shape[0],
            num_classes=y_train.shape[1],
            vocab_sizes=[len(words_vocab_processor.vocabulary_), len(tags_vocab_processor.vocabulary_), len(labels_vocab_processor.vocabulary_)],
            embedding_sizes=[embedding_dimension, 4, 4],
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        var_summaries = []
        for var in tf.trainable_variables():
            var_hist_summary = tf.summary.histogram("{}/var/hist".format(var.name), var)
            var_summaries.append(var_hist_summary)
        var_summaries_merged = tf.summary.merge(var_summaries)

        # Output directory for models and summaries
        timestamp = str(datetime.datetime.now()) + ",glove,init-embeddings-curr-weight-nontrainable,fancy-inits,label-weights-per-edge-per-dim,6-layers,word-plus-label-plus-tag-embedding,p-c-p-child-label-p-tag-embedding,fc"
        out_dir = os.path.abspath(os.path.join("/u/a/n/anant/539_project", "runs", timestamp))
        # out_dir = "/u/a/n/anant/539_project/runs/2017-12-05 15:45:58.599020,glove,init-embeddings-curr-weight-nontrainable,label-weights-per-edge-per-dim,4-layers,word-plus-label-embedding,parent-child-parent-child-label-embedding"
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        # debug_summary = tf.summary.tensor_summary("debug_info", cnn.debug_info)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, var_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        # Best checkpoints directory
        best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "best_checkpoints"))
        best_checkpoint_prefix = os.path.join(best_checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabularies
        words_vocab_processor.save(os.path.join(out_dir, "vocab"))
        tags_vocab_processor.save(os.path.join(out_dir, "tags_vocab"))
        labels_vocab_processor.save(os.path.join(out_dir, "labels_vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = words_vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
            sess.run(cnn.W.assign(initW))

        def train_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, learning_rate):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_words: x_words_batch,
              cnn.input_tags: x_tags_batch,
              cnn.input_labels: x_labels_batch,
              cnn.input_indices: x_indices_batch,
              cnn.input_trees: x_trees_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.learning_rate: learning_rate
              # cnn.seq: x_words_batch.shape[1]
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g},"
                  .format(time_str, step, loss, accuracy, learning_rate))
            print(str(len(x_words_batch)) + " " + str(len(x_words_batch[0])) + " ")
            print(cnn.input_words)
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_words: x_words_batch,
              cnn.input_tags: x_tags_batch,
              cnn.input_labels: x_labels_batch,
              cnn.input_indices: x_indices_batch,
              cnn.input_trees: x_trees_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
              # cnn.seq: x_words_batch.shape[1]
            }
            print('burrahhh', file=sys.stderr)
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy


        x_words_dev, x_tags_dev, x_labels_dev, x_indices_dev, x_trees_dev = zip(*x_dev)
        x_words_dev = np.array(x_words_dev)
        x_tags_dev = np.array(x_tags_dev)
        x_labels_dev = np.array(x_labels_dev)
        x_indices_dev = np.array(x_indices_dev)
        x_trees_dev = list(x_trees_dev)
        x_trees_dev2 = np.zeros([x_words_dev.shape[0], x_words_dev.shape[1], x_words_dev.shape[1]])
        for i in range(len(x_trees_dev)):
            bla = eval(x_trees_dev[i])
            x_trees_dev2[i,0:len(bla),0:len(bla)] = bla
        # x_trees_batch = np.array(x_trees_batch)
        x_trees_dev = x_trees_dev2
        # x_trees_dev = np.reshape(x_trees_dev, (np.shape(x_words_dev)[0], np.shape(x_words_dev)[1], np.shape(x_words_dev)[1]))

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # It uses dynamic learning rate with a high value at the beginning to speed up the training
        max_learning_rate = 0.005
        min_learning_rate = 0.002
        decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
        # Training loop. For each batch...
        counter = 0
        max_accuracy = 0.0
        for batch in batches:
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            x_batch, y_batch = zip(*batch)
            x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch = zip(*x_batch)
            # print(np.shape(x_trees_batch))
            x_words_batch = np.array(x_words_batch)
            # print(np.shape(x_words_batch))
            x_tags_batch = np.array(x_tags_batch)
            # print(np.shape(x_tags_batch))
            x_labels_batch = np.array(x_labels_batch)
            # print(np.shape(x_labels_batch))
            x_indices_batch = np.array(x_indices_batch)
            # print(np.shape(x_indices_batch))
            x_trees_batch = list(x_trees_batch)
            x_trees_batch2 = np.zeros([x_words_batch.shape[0], x_words_batch.shape[1], x_words_batch.shape[1]])
            for i in range(len(x_trees_batch)):
                bla = eval(x_trees_batch[i])
                x_trees_batch2[i,0:len(bla),0:len(bla)] = bla
            # x_trees_batch = np.array(x_trees_batch)
            x_trees_batch = x_trees_batch2
            # print(np.shape(x_trees_batch))

            # x_trees_batch = np.reshape(x_trees_batch, (np.shape(x_words_batch)[0], np.shape(x_words_batch)[1], np.shape(x_words_batch)[1]))
            train_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, learning_rate)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                accuracy = dev_step(x_words_dev, x_tags_dev, x_labels_dev, x_indices_dev, x_trees_dev, y_dev, writer=dev_summary_writer)
                accuracy = float(accuracy)
                print("")
                print("accuracy:" + str(accuracy))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    path = saver.save(sess, best_checkpoint_prefix, global_step=current_step)
                    print("Saved best model checkpoint to {}\n".format(path))

            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))
