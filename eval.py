#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import itertools

preps = ['at', 'on', 'in', 'by', 'for', 'against', 'to', 'from', 'between', 'during', 'with', 'about', 'of']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/u/a/n/anant/539_project/runs/2017-12-10 17:11:50.923482,glove,baseline,fc-3-layer,quadruple-hidden-neurons/best_checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

x_words_raw, x_tags, x_labels, x_trees, x_indices, y, y_labels = data_helpers.load_data_labels('/u/a/n/anant/Dropbox/539_project/generated_test_data/')
x_words = x_words_raw
# x_words = x_words[1:1000]
# x_tags = x_tags[1:1000]
# x_labels = x_labels[1:1000]
# x_trees = x_trees[1:1000]
# x_indices = x_indices[1:1000]
# y_labels = y_labels[1:1000]
max_document_length = 50
valid_indices = []
for i in range(len(x_words)):
    if len(x_words[i].split(" ")) <= max_document_length:
        valid_indices.append(i)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_words = np.array(list(vocab_processor.transform(x_words)))

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "tags_vocab")
tags_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_tags = np.array(list(tags_vocab_processor.transform(x_tags)))

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "labels_vocab")
labels_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_labels = np.array(list(labels_vocab_processor.transform(x_labels)))
        
for i in range(max(max_document_length, len(x_words))):
    if x_indices[i] < max_document_length:
        x_words[i][int(x_indices[i])] = 0
x_indices = np.array(x_indices)
x_trees = np.array(x_trees)
# x_trees = x_trees.reshape(len(x_words), -1)
x_feats = (list(zip(x_words, x_tags, x_labels, x_indices, x_trees)))

x_feats = np.array([x_feats[i] for i in valid_indices])
x_words = np.array([x_words[i] for i in valid_indices])
x_words_raw = np.array([x_words_raw[i] for i in valid_indices])
y_labels = np.array([y_labels[i] for i in valid_indices])
y_test = y_labels

print("\nEvaluating...\n")

# Evaluation
# ==================================================
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir[:-11] + "best_checkpoints")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# checkpoint_file = "/u/a/n/anant/539_project/runs/2017-12-10 19:01:26.103352,glove,init-embeddings-curr-weight-nontrainable,fancy-inits,label-weights-per-edge-per-dim,6-layers,word-plus-label-plus-tag-embedding,p-c-p-child-label-p-tag-embedding,fc/best_checkpoints/model-9700"
print("checkpoint file: " + checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_words = graph.get_operation_by_name("input_words").outputs[0]
        input_tags = graph.get_operation_by_name("input_tags").outputs[0]
        input_labels = graph.get_operation_by_name("input_labels").outputs[0]
        input_indices = graph.get_operation_by_name("input_indices").outputs[0]
        input_trees = graph.get_operation_by_name("input_trees").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(zip(x_feats, y)), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            x_batch, y_batch = zip(*x_test_batch)
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

            feed_dict = {
              input_words: x_words_batch,
              input_tags: x_tags_batch,
              input_labels: x_labels_batch,
              input_indices: x_indices_batch,
              input_trees: x_trees_batch,
              input_y: y_batch,
              dropout_keep_prob: 1.0
              # cnn.seq: x_words_batch.shape[1]
            }
            batch_predictions_scores = sess.run([predictions, scores], feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities

# Print accuracy if y_test is defined
if y_test is not None:
    print(y_test)
    print(all_predictions)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions, target_names=['at', 'on', 'in', 'by', 'for', 'against', 'to', 'from', 'between', 'during', 'with', 'about', 'of']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
# print(x_words.shape)
# print(len(all_predictions))
predictions_human_readable = np.column_stack((x_words_raw,
                                              [preps[int(prediction)] for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
