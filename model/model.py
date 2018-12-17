import tensorflow as tf
from utils.conv import causal_conv, wave_net_activation, channel_normalization
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf


class TCN(object):
    def __init__(self, configs, input_ph, is_training):
        self.nb_filters = configs.nb_filters
        self.kernel_size = configs.kernel_size
        self.nb_stacks = configs.nb_stacks
        self.dilations = list(map(int, configs.dilations.split(",")))
        self.activation = configs.activation
        self.use_skip_connections = configs.use_skip_connections
        self.dropout_rate = configs.dropout_rate
        self.reture_sequences = configs.return_sequences
        self.max_len = configs.max_len
        self.num_class = configs.num_classes
        self.batch_size = configs.batch_size
        self.vocab_size = configs.vocab_size
        self.embed_size = configs.embed_size
        self.epochs = configs.epochs
        self.learning_rate = configs.learning_rate
        # self.save_dir = configs.save_dir
        # self.output_folder = configs.output_folder
        self.logits = None
        # self.checkpoint = configs.checkpoint
        self.input_ph = input_ph
        self.is_training = is_training
        # self.num_labels = configs.num_labels
        self.type = configs.type

        self.build_network()


    def residual_block(self, inputs,
                       index_stack,
                       dilation,
                       nb_filters,
                       kernel_size,
                       dropout_rate=0.0,
                       activation="relu",
                       is_training=True):
        original_x = inputs
        with tf.variable_scope("residual_block", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("dilated_causal_conv_1"):
                filter_shape = [kernel_size, nb_filters, nb_filters]
                x = causal_conv(inputs, filter_shape, dilation)

            with tf.variable_scope("layer_norm_1"):
                x = tf.contrib.layers.layer_norm(x)

            with tf.variable_scope("activation_1"):
                if activation == "norm_relu":
                    x = tf.nn.relu(x)
                    x = channel_normalization(x)
                elif activation == "wavenet":
                    x = wave_net_activation(x)
                else:
                    x = tf.nn.relu(x)

            with tf.variable_scope("dropout_1"):
                x = tf.contrib.layers.dropout(x, keep_prob=dropout_rate,
                                              noise_shape=[1,1,nb_filters], is_training=is_training)

            with tf.variable_scope("dilated_causal_conv_2"):
                filter_shape = [kernel_size, nb_filters, nb_filters]
                x = causal_conv(x, filter_shape, dilation)

            with tf.variable_scope("layer_norm_2"):
                x = tf.contrib.layers.layer_norm(x)

            with tf.variable_scope("activation_2"):
                if activation == "norm_relu":
                    x = tf.nn.relu(x)
                    x = channel_normalization(x)
                elif activation == "wavenet":
                    x = wave_net_activation(x)
                else:
                    x = tf.nn.relu(x)

            with tf.variable_scope("dropout_2"):
                x = tf.contrib.layers.dropout(x, keep_prob=dropout_rate,
                                              noise_shape=[1, 1, nb_filters],
                                              is_training=is_training)
            original_x = tf.layers.Conv1D(filters=nb_filters, kernel_size=1)(original_x)
        res_x = tf.add(original_x, x)
        return res_x, x

    def process_dilations(self, dilations):
        def is_power_of_two(num):
            return num != 0 and ((num & (num - 1)) == 0)

        if all([is_power_of_two(i) for i in dilations]):
            return dilations
        else:
            new_dilations = [2 ** i for i in range(len(dilations))]
            print('Updated dilations from ',
                  dilations, 'to', new_dilations)
            return new_dilations
        # if self.type == "classification":
        #     self.loss = tf.reduce_mean(self.get_loss(self.logits, self.label_ph))
        # elif self.type == 'ner':
        #     self.loss, _ = self.get_loss(self.logits, self.label_ph)
        #
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=tf.train.global_step())
    def get_loss(self, logits, labels, lengths=None):

        self.lengths = lengths
        if self.type == 'classification':
            with tf.variable_scope("classification_loss", reuse=tf.AUTO_REUSE):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        elif self.type == 'ner':
            with tf.variable_scope("ner", reuse=tf.AUTO_REUSE):
                self.trans = tf.get_variable("transitions",
                                        shape=[self.num_class, self.num_class],
                                        initializer=initializers.xavier_initializer())
                log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=labels,
                    transition_params=self.trans,
                    sequence_lengths=lengths)
                self.loss = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=tf.train.get_global_step())

    def build_network(self):
        with tf.variable_scope("embedding_scope"):
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size],
                                        initializer=tf.random_uniform_initializer(minval = -0.5, maxval = 0.5))
            input_embed = tf.nn.embedding_lookup(embedding, self.input_ph, name="input_embed")

        x = tf.layers.Conv1D(filters=self.nb_filters, kernel_size=1, padding="valid")(input_embed)
        with tf.variable_scope("resnet"):
            skip_connections = []
            for i, d in enumerate(self.dilations):
                x, skip_out = self.residual_block(inputs=x, index_stack=i, dilation=d,
                                                  nb_filters=self.nb_filters, kernel_size=self.kernel_size,
                                                  dropout_rate=self.dropout_rate, is_training=self.is_training)
                skip_connections.append(skip_out)
            if self.use_skip_connections:
                x = tf.add_n(skip_connections) + x
            x = tf.nn.relu(x)

        if not self.reture_sequences:
            output_slice_index = -1
            x = x[:, output_slice_index, :]

        if self.type == "classification":
            with tf.variable_scope("fully_connected_1"):
                in_dim = x.get_shape().as_list()[-1]
                w = tf.get_variable(name="w", shape=[in_dim, 20], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", shape=[20], initializer=tf.constant_initializer(0.0))
                x = tf.nn.relu(tf.matmul(x, w) + b)

            with tf.variable_scope("fully_connected_2"):
                in_dim = x.get_shape().as_list()[-1]
                w = tf.get_variable(name="w", shape=[20, self.num_class],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", shape=[self.num_class], initializer=tf.constant_initializer(0.0))
                self.logits = tf.matmul(x, w) + b  # [N, 2]
        elif self.type == "ner":
            with tf.variable_scope("fully_connected_1"):
                in_dim = x.get_shape().as_list()[-1]
                w = tf.get_variable(name="w", shape=[in_dim, 100], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", shape=[100], initializer=tf.constant_initializer(0.0))
                x = tf.nn.relu(tf.matmul(x, w) + b)

            with tf.variable_scope("fully_connected_2"):
                in_dim = x.get_shape().as_list()[-1]
                w = tf.get_variable(name="w", shape=[100, self.max_len],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", shape=[self.max_len], initializer=tf.constant_initializer(0.0))
                self.logits = tf.matmul(x, w) + b  # [N, 2]

    def get_metrics(self, logits):
        print("==============================")
        print(logits)
        print("==============================")
        if self.type == "classification":
            self.predictions = tf.squeeze(tf.argmax(self.logits, axis = -1), name = "predictions")
        elif self.type == "ner":
            self.predictions, _ = crf.crf_decode(potentials=logits, transition_params=self.trans, sequence_length=self.lengths)
        # self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)


