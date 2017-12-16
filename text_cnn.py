import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_sizes,
      embedding_sizes, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        print("seq")
        print(sequence_length)
        # self.seq = tf.placeholder(tf.int32, [1], name="seq")
        self.input_words = tf.placeholder(tf.int32, [None, None], name="input_words")
        self.input_tags = tf.placeholder(tf.int32, [None, None], name="input_tags")
        self.input_labels = tf.placeholder(tf.int32, [None, None], name="input_labels")
        self.input_indices = tf.placeholder(tf.int32, [None], name="input_indices")
        self.input_trees = tf.placeholder(tf.int32, [None, None, None], name="input_trees")
        batch_size = tf.shape(self.input_words)[0]
        # self.input_x = tf.Print(self.input_x, [tf.shape(self.input_x)], 'blah blah')
        # print(tf.shape(self.input_x))
        # self.debug_info = tf.shape(self.input_words)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_sizes[0], embedding_sizes[0]], -1.0, 1.0),
                name="W_chars", trainable=False)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_words)

            self.W_tags = tf.Variable(
                tf.random_uniform([vocab_sizes[1], embedding_sizes[1]], -1.0, 1.0),
                name="W_tags")
            self.embedded_tags = tf.nn.embedding_lookup(self.W_tags, self.input_tags)

            self.W_labels = tf.Variable(
                tf.random_uniform([vocab_sizes[2], embedding_sizes[2]], -1.0, 1.0),
                name="W_labels")
            self.embedded_labels = tf.nn.embedding_lookup(self.W_labels, self.input_labels)

        # Parent labels (used to calculate weights in next layer)
        with tf.name_scope("embedded-parents"):
            parent_trees = tf.equal(self.input_trees, [-1])
            dont_have_parents = tf.logical_not(tf.reduce_any(parent_trees, axis=2, keep_dims=True))
            parent_trees = tf.concat([parent_trees, dont_have_parents], axis=2)
            parent_trees = tf.cast(parent_trees, tf.float32)
            # self.parent_indices = tf.where(parent_trees)
            shape = tf.shape(self.embedded_chars)
            concated_embedded_chars = tf.concat([self.embedded_chars, tf.zeros([shape[0], 1, embedding_sizes[0]])], axis=1)
            concated_embedded_labels = tf.concat([self.embedded_labels, tf.zeros([shape[0], 1, embedding_sizes[2]])], axis=1)
            # self.embedded_parent_chars = tf.matmul(parent_trees, concated_embedded_chars)
            self.embedded_parent_labels = tf.matmul(parent_trees, concated_embedded_labels)

        with tf.name_scope("weight-from-labels"):
            W = tf.Variable(tf.truncated_normal([embedding_sizes[2], embedding_sizes[0]], stddev=0.1), name="W")
            # W = tf.Print(W, [W], "weight-from-labels: ")
            b = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b")
            self.label_weights = tf.tensordot(self.embedded_labels, W, [[-1], [0]])
            self.label_weights = tf.add(self.label_weights, b)
            W_p2c = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[0]], stddev=0.1), name="W_p2c")
            b_label_edge_p2c = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b_label_edge_p2c")
            self.label_edge_weights_p2c = tf.tensordot(tf.concat([self.embedded_parent_labels, self.embedded_labels], axis=2), W_p2c, [[-1], [0]])
            self.label_edge_weights_p2c = tf.add(self.label_edge_weights_p2c, b_label_edge_p2c)
            W_c2p = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[0]], stddev=0.1), name="W_c2p")
            b_label_edge_c2p = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b_label_edge_c2p")
            self.label_edge_weights_c2p = tf.tensordot(tf.concat([self.embedded_labels, self.embedded_parent_labels], axis=2), W_c2p, [[-1], [0]])
            self.label_edge_weights_c2p = tf.add(self.label_edge_weights_c2p, b_label_edge_c2p)
            # self.label_edge_weights_c2p = tf.Print(self.label_edge_weights_c2p, [tf.norm(self.label_edge_weights_c2p)], "norm label_edge_weights_c2p:")
            # self.label_edge_weights_p2c = tf.Print(self.label_edge_weights_p2c, [tf.norm(self.label_edge_weights_p2c)], "norm label_edge_weights_p2c:")
            # self.label_weights = tf.Print(self.label_weights, [tf.norm(self.label_weights)], "norm label_weights:")

            # W = tf.Variable(tf.truncated_normal([embedding_sizes[0], embedding_sizes[0]+embedding_sizes[2]], stddev=0.1), name="W")
            # # W = tf.Print(W, [W], "weight-from-labels: ")
            # b = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b")
            # tf.nn.xw_plus_b(tf.concat([self.parent_embeddings, self.parent_tag_embeddings, self.child_tag_embeddings, self.child_embeddings, self.child_label_embeddings, self.parent_label_embeddings], axis=1), W, b)
            # self.label_weights = tf.tensordot(self.embedded_labels, W, [[-1], [0]])
            # self.label_weights = tf.add(self.label_weights, b)
            # W_p2c = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[0]], stddev=0.1), name="W_p2c")
            # b_label_edge_p2c = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b_label_edge_p2c")
            # self.label_edge_weights_p2c = tf.tensordot(tf.concat([self.embedded_parent_labels, self.embedded_labels], axis=2), W_p2c, [[-1], [0]])
            # self.label_edge_weights_p2c = tf.add(self.label_edge_weights_p2c, b_label_edge_p2c)
            # W_c2p = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[0]], stddev=0.1), name="W_c2p")
            # b_label_edge_c2p = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[0]]), name="b_label_edge_c2p")
            # self.label_edge_weights_c2p = tf.tensordot(tf.concat([self.embedded_labels, self.embedded_parent_labels], axis=2), W_c2p, [[-1], [0]])
            # self.label_edge_weights_c2p = tf.add(self.label_edge_weights_c2p, b_label_edge_c2p)

        with tf.name_scope("weight-from-labels-tags"):
            W = tf.Variable(tf.truncated_normal([embedding_sizes[2], embedding_sizes[1]], stddev=0.1), name="W")
            # W = tf.Print(W, [W], "weight-from-labels-tags: ")
            b = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[1]]), name="b")
            self.label_tag_weights = tf.tensordot(self.embedded_labels, W, [[-1], [0]])
            self.label_tag_weights = tf.add(self.label_tag_weights, b)
            W_p2c = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[1]], stddev=0.1), name="W_p2c")
            b_label_edge_p2c = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[1]]), name="b_label_edge_p2c")
            self.label_tag_edge_weights_p2c = tf.tensordot(tf.concat([self.embedded_parent_labels, self.embedded_labels], axis=2), W_p2c, [[-1], [0]])
            self.label_tag_edge_weights_p2c = tf.add(self.label_tag_edge_weights_p2c, b_label_edge_p2c)
            W_c2p = tf.Variable(tf.truncated_normal([2*embedding_sizes[2], embedding_sizes[1]], stddev=0.1), name="W_c2p")
            b_label_edge_c2p = tf.Variable(tf.constant(0.1, shape=[embedding_sizes[1]]), name="b_label_edge_c2p")
            self.label_tag_edge_weights_c2p = tf.tensordot(tf.concat([self.embedded_labels, self.embedded_parent_labels], axis=2), W_c2p, [[-1], [0]])
            self.label_tag_edge_weights_c2p = tf.add(self.label_tag_edge_weights_c2p, b_label_edge_c2p)
            # self.label_tag_edge_weights_c2p = tf.Print(self.label_tag_edge_weights_c2p, [tf.norm(self.label_tag_edge_weights_c2p)], "norm label_tag_edge_weights_c2p:")
            # self.label_tag_edge_weights_p2c = tf.Print(self.label_tag_edge_weights_p2c, [tf.norm(self.label_tag_edge_weights_p2c)], "norm label_tag_edge_weights_p2c:")
            # self.label_tag_weights = tf.Print(self.label_tag_weights, [tf.norm(self.label_tag_weights)], "norm label_tag_weights:")

        tree_weighted_chars = self.graph_layer('layer1', self.embedded_chars, embedding_sizes, 1)
        tree_weighted_chars = self.graph_layer('layer2', tree_weighted_chars, embedding_sizes, 2)
        tree_weighted_chars = self.graph_layer('layer3', tree_weighted_chars, embedding_sizes, 3)
        tree_weighted_chars = self.graph_layer('layer4', tree_weighted_chars, embedding_sizes, 4)
        tree_weighted_chars = self.graph_layer('layer5', tree_weighted_chars, embedding_sizes, 5)
        tree_weighted_chars = self.graph_layer('layer6', tree_weighted_chars, embedding_sizes, 6)
        # tree_weighted_chars = self.graph_layer('layer7', tree_weighted_chars, embedding_sizes, 7)
        # tree_weighted_chars = self.graph_layer('layer8', tree_weighted_chars, embedding_sizes, 8)
        # tree_weighted_chars = self.graph_layer('layer9', tree_weighted_chars, embedding_sizes, 9)
        # tree_weighted_chars = self.graph_layer('layer10', tree_weighted_chars, embedding_sizes, 10)
        # tree_weighted_chars = self.embedded_chars

        tree_weighted_tags = self.graph_tag_layer('tag-layer1', self.embedded_tags, embedding_sizes, 1)
        tree_weighted_tags = self.graph_tag_layer('tag-layer2', tree_weighted_tags, embedding_sizes, 2)
        tree_weighted_tags = self.graph_tag_layer('tag-layer3', tree_weighted_tags, embedding_sizes, 3)
        tree_weighted_tags = self.graph_tag_layer('tag-layer4', tree_weighted_tags, embedding_sizes, 4)
        tree_weighted_tags = self.graph_tag_layer('tag-layer5', tree_weighted_tags, embedding_sizes, 5)
        tree_weighted_tags = self.graph_tag_layer('tag-layer6', tree_weighted_tags, embedding_sizes, 6)
        # tree_weighted_tags = self.graph_tag_layer('tag-layer7', tree_weighted_tags, embedding_sizes, 7)
        # tree_weighted_tags = self.graph_tag_layer('tag-layer8', tree_weighted_tags, embedding_sizes, 8)
        # tree_weighted_tags = self.graph_tag_layer('tag-layer9', tree_weighted_tags, embedding_sizes, 9)
        # tree_weighted_tags = self.graph_tag_layer('tag-layer10', tree_weighted_tags, embedding_sizes, 10)
        # tree_weighted_tags = self.embedded_tags

        
        with tf.name_scope("final-prep-embeddings-slicing"):
            # self.label_weighted_chars3 = tf.multiply(self.tree_weighted_chars2, label_weights)
            # self.tree_weighted_chars3 = tf.matmul(tree_weights, self.label_weighted_chars3)
            # self.label_weighted_chars4 = tf.multiply(self.tree_weighted_chars3, label_weights)
            # self.tree_weighted_chars4 = tf.matmul(tree_weights, self.label_weighted_chars4)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            prep_row_indices = tf.stack([tf.range(batch_size), self.input_indices])
            self.prep_embeddings = tf.gather_nd(tree_weighted_chars, tf.transpose(prep_row_indices))
            prep_row_slices = tf.gather_nd(self.input_trees, tf.transpose(prep_row_indices))
            parent_row_matrix = tf.equal(prep_row_slices, [-1])
            # dont_have_parents = tf.logical_not(tf.reduce_any(parent_row_matrix, axis=1, keep_dims=True))
            # parent_row_matrix = tf.concat([parent_row_matrix, dont_have_parents], axis=1)
            # parent_row_indices = tf.where(parent_row_matrix)
            # shape = tf.shape(self.tree_weighted_chars2)
            # concated_tree_weighted_chars = tf.concat([self.tree_weighted_chars2, tf.zeros([shape[0], 1, embedding_sizes[0]])], axis=1)
            # concated_tree_weighted_labels = tf.concat([self.embedded_labels, tf.zeros([shape[0], 1, embedding_sizes[2]])], axis=1)

            # self.parent_embeddings = tf.gather_nd(concated_tree_weighted_chars, parent_row_indices)
            # self.parent_label_embeddings = tf.gather_nd(concated_tree_weighted_labels, parent_row_indices)

            parent_row_matrix = tf.cast(parent_row_matrix, tf.float32)
            parent_row_matrix = tf.expand_dims(parent_row_matrix, 1)
            self.parent_embeddings = tf.matmul(parent_row_matrix, tree_weighted_chars)
            self.parent_embeddings = tf.squeeze(self.parent_embeddings)
            self.parent_label_embeddings = tf.matmul(parent_row_matrix, self.embedded_labels)
            self.parent_label_embeddings = tf.squeeze(self.parent_label_embeddings)
            self.parent_tag_embeddings = tf.matmul(parent_row_matrix, tree_weighted_tags)
            self.parent_tag_embeddings = tf.squeeze(self.parent_tag_embeddings)

            # child_row_matrix = tf.equal(prep_row_slices, [1])
            # dont_have_child = tf.logical_not(tf.reduce_any(child_row_matrix, axis=1, keep_dims=True))
            # child_row_matrix = tf.concat([child_row_matrix, dont_have_child], axis=1)
            # child_row_indices = tf.where(child_row_matrix)
            # self.child_embeddings = tf.gather_nd(concated_tree_weighted_chars, child_row_indices)
            # self.child_label_embeddings = tf.gather_nd(concated_tree_weighted_labels, child_row_indices)

            child_row_matrix = tf.equal(prep_row_slices, [1])
            child_row_matrix = tf.cast(child_row_matrix, tf.float32)
            child_row_matrix = tf.expand_dims(child_row_matrix, 1)
            self.child_embeddings = tf.matmul(child_row_matrix, tree_weighted_chars)
            self.child_embeddings = tf.squeeze(self.child_embeddings)
            self.child_label_embeddings = tf.matmul(child_row_matrix, self.embedded_labels)
            self.child_label_embeddings = tf.squeeze(self.child_label_embeddings)
            self.child_tag_embeddings = tf.matmul(child_row_matrix, tree_weighted_tags)
            self.child_tag_embeddings = tf.squeeze(self.child_tag_embeddings)

            # self.prep_embeddings = tf.Print(self.prep_embeddings, [self.prep_embeddings], "prep_embeddings: ", summarize=100)



        # # Create a convolution + maxpool layer for each filter size
        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size, embedding_sizes[0], 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         conv = tf.nn.conv2d(
        #             self.embedded_chars_expanded,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs.append(pooled)

        # # Combine all the pooled features
        # num_filters_total = num_filters * len(filter_sizes)
        # self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # # Add dropout
        # with tf.name_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # # Concat layer
        # with tf.name_scope("concat-features"):
        #     self.concated_features = tf.concat([self.prep_embeddings, self.tag_embeddings])

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[2*embedding_sizes[0] + 2*embedding_sizes[2] + 2*embedding_sizes[1], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Print(W, [W], "output_weight: ")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(tf.concat([self.parent_embeddings, self.parent_tag_embeddings, self.child_tag_embeddings, self.child_embeddings, self.child_label_embeddings, self.parent_label_embeddings], axis=1), W, b, name="scores")
            self.scores = tf.tanh(self.scores)
            W2 = tf.get_variable(
                "W2_output",
                shape=[num_classes, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.Print(W2, [W2], "output_weight_2: ")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            self.scores = tf.nn.xw_plus_b(self.scores, W2, b2)
            # self.scores = tf.tanh(self.scores)
            # W3 = tf.get_variable(
            #     "W3_output",
            #     shape=[num_classes*2, num_classes*2],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # b3 = tf.Variable(tf.constant(0.1, shape=[num_classes*2]), name="b3")
            # self.scores = tf.nn.xw_plus_b(self.scores, W3, b3)
            # self.scores = tf.tanh(self.scores)
            # W4 = tf.get_variable(
            #     "W4_output",
            #     shape=[num_classes*2, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")
            # self.scores = tf.nn.xw_plus_b(self.scores, W4, b4)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.loss = tf.Print(self.loss, [self.loss], "loss_debug:", summarize=100)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def graph_layer(self, scope, prev_chars, embedding_sizes, layer_number):
        with tf.name_scope(scope+"weight-from-trees") as scope:
            label_c2p_weighted_chars = tf.multiply(prev_chars, self.label_edge_weights_c2p)
            label_p2c_weighted_chars = tf.multiply(prev_chars, self.label_edge_weights_p2c)
            label_weighted_chars = tf.multiply(prev_chars, self.label_weights)
            # label_c2p_weighted_chars /= tf.norm(self.label_edge_weights_c2p)
            # label_p2c_weighted_chars /= tf.norm(self.label_edge_weights_p2c)
            # label_weighted_chars /= tf.norm(self.label_weights)
            # self.input_trees2 = tf.Print(self.input_trees, [self.input_trees[0][0]], "input_trees: ")
            blah = (layer_number != 1)
            parent_weight =tf.Variable(tf.constant(0.1*layer_number, shape=[1]), name="parent_weight", trainable=blah)
            curr_weight =tf.Variable(tf.constant(1.0, shape=[1]), name="curr_weight", trainable=False)
            child_weight =tf.Variable(tf.constant(0.1*layer_number, shape=[1]), name="child_weight", trainable=blah)
            parent_weight = tf.Print(parent_weight, [parent_weight], scope+"parent_weight_1:")
            curr_weight = tf.Print(curr_weight, [curr_weight], scope+"curr_weight_1:")
            child_weight = tf.Print(child_weight, [child_weight], scope+"child_weight_1:")
            parent_equal = tf.cast(tf.equal(self.input_trees, tf.constant([-1])), tf.float32)
            parent_weights = parent_equal * parent_weight
            # tree_weights = tf.eye(tf.shape(self.input_trees)[1]) * curr_weight
            child_equal = tf.cast(tf.equal(self.input_trees, tf.constant([1])), tf.float32)
            child_weights = child_equal * child_weight
            # tree_weights = tf.Print(tree_weights, [tree_weights[0][0]], "tree_weights: ")
            tree_weighted_chars = curr_weight * label_weighted_chars
            tree_weighted_chars += tf.matmul(parent_weights, label_p2c_weighted_chars)
            tree_weighted_chars += tf.matmul(child_weights, label_c2p_weighted_chars)
            # total_normalization_constants = tf.norm(tf.matmul(parent_weights, self.label_edge_weights_p2c), axis=-1, keep_dims=True) + tf.norm(tf.matmul(child_weights, self.label_edge_weights_c2p), axis=-1, keep_dims=True) + tf.norm(curr_weight*self.label_weights, axis=-1, keep_dims=True) + 1e-30
            # total_normalization_constants = tf.matmul(parent_weights, self.label_edge_weights_p2c) + tf.matmul(child_weights, self.label_edge_weights_c2p) + curr_weight*self.label_weights + 1e-30
            # total_normalization_constants = tf.Print(total_normalization_constants, [total_normalization_constants[0,:,:]], "total_normalization_constants:", summarize=100)
            # normalization_constants = tf.reduce_sum(parent_weights, axis=-1, keep_dims=True) + tf.reduce_sum(child_weights, axis=-1, keep_dims=True) + curr_weight
            # tree_weighted_chars = tf.divide(tree_weighted_chars, normalization_constants)
            # tree_weighted_chars = tf.divide(tree_weighted_chars, total_normalization_constants)
            # tree_weighted_chars = tf.Print(tree_weighted_chars, [tree_weighted_chars], scope+"tree_weighted_chars:")
        
        return tree_weighted_chars

    def graph_tag_layer(self, scope, prev_tags, embedding_sizes, layer_number):
        with tf.name_scope(scope+"weight-from-trees") as scope:
            label_c2p_weighted_chars = tf.multiply(prev_tags, self.label_tag_edge_weights_c2p)
            label_p2c_weighted_chars = tf.multiply(prev_tags, self.label_tag_edge_weights_p2c)
            label_weighted_chars = tf.multiply(prev_tags, self.label_tag_weights)
            # label_c2p_weighted_chars /= tf.norm(self.label_edge_weights_c2p)
            # label_p2c_weighted_chars /= tf.norm(self.label_edge_weights_p2c)
            # label_weighted_chars /= tf.norm(self.label_weights)
            # self.input_trees2 = tf.Print(self.input_trees, [self.input_trees[0][0]], "input_trees: ")
            blah = (layer_number != 1)
            parent_weight =tf.Variable(tf.constant(0.1*layer_number, shape=[1]), name="parent_weight", trainable=blah)
            curr_weight =tf.Variable(tf.constant(1.0, shape=[1]), name="curr_weight", trainable=False)
            child_weight =tf.Variable(tf.constant(0.1*layer_number, shape=[1]), name="child_weight", trainable=blah)
            parent_weight = tf.Print(parent_weight, [parent_weight], scope+"parent_weight_1:")
            curr_weight = tf.Print(curr_weight, [curr_weight], scope+"curr_weight_1:")
            child_weight = tf.Print(child_weight, [child_weight], scope+"child_weight_1:")
            parent_equal = tf.cast(tf.equal(self.input_trees, tf.constant([-1])), tf.float32)
            parent_weights = parent_equal * parent_weight
            # tree_weights = tf.eye(tf.shape(self.input_trees)[1]) * curr_weight
            child_equal = tf.cast(tf.equal(self.input_trees, tf.constant([1])), tf.float32)
            child_weights = child_equal * child_weight
            # tree_weights = tf.Print(tree_weights, [tree_weights[0][0]], "tree_weights: ")
            tree_weighted_chars = curr_weight * label_weighted_chars
            tree_weighted_chars += tf.matmul(parent_weights, label_p2c_weighted_chars)
            tree_weighted_chars += tf.matmul(child_weights, label_c2p_weighted_chars)
            # total_normalization_constants = tf.norm(tf.matmul(parent_weights, self.label_edge_weights_p2c), axis=-1, keep_dims=True) + tf.norm(tf.matmul(child_weights, self.label_edge_weights_c2p), axis=-1, keep_dims=True) + tf.norm(curr_weight*self.label_weights, axis=-1, keep_dims=True) + 1e-30
            # total_normalization_constants = tf.matmul(parent_weights, self.label_edge_weights_p2c) + tf.matmul(child_weights, self.label_edge_weights_c2p) + curr_weight*self.label_weights + 1e-30
            # total_normalization_constants = tf.Print(total_normalization_constants, [total_normalization_constants[0,:,:]], "total_normalization_constants:", summarize=100)
            # normalization_constants = tf.reduce_sum(parent_weights, axis=-1, keep_dims=True) + tf.reduce_sum(child_weights, axis=-1, keep_dims=True) + curr_weight
            # tree_weighted_chars = tf.divide(tree_weighted_chars, normalization_constants)
            # tree_weighted_chars = tf.divide(tree_weighted_chars, total_normalization_constants)
            # tree_weighted_chars = tf.Print(tree_weighted_chars, [tree_weighted_chars], scope+"tree_weighted_chars:")
        
        return tree_weighted_chars