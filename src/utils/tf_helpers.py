import os, sys
import signal
import math, random
import shutil
from sklearn.metrics import *
import tensorflow as tf

#--------------------------------------------------------------
# Model helpers
#--------------------------------------------------------------
def variable_summaries(var, scope_name, name):
    """Attach summaries to a Tensor."""
    var_summaries = []
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        var_summaries.append(tf.scalar_summary(scope_name + '/mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        var_summaries.append(tf.scalar_summary(scope_name + '/sttdev', stddev))
        #var_summaries.append(tf.scalar_summary(scope_name + '/max', tf.reduce_max(var)))
        #var_summaries.append(tf.scalar_summary(scope_name + '/min', tf.reduce_min(var)))
        var_summaries.append(tf.histogram_summary(scope_name + '/' + name, var))
    return var_summaries

def savemodel(session, step, saver, checkpoint_dir, run_name, posfix=""):
    sys.stdout.write('Saving model...')
    sys.stdout.flush()
    save_path = saver.save(session, checkpoint_dir + '/' + run_name + posfix, global_step=step + 1)
    print('[Done]')
    return save_path

def projection_weights(orig_size, target_size, name=''):
    weight = tf.Variable(tf.truncated_normal([orig_size, target_size], stddev=1.0 / math.sqrt(target_size)), name=name + 'weight')
    bias = tf.Variable(tf.zeros([target_size]), name=name + 'bias')
    print 'Projection weights: %s' % name
    return weight, bias

def get_projection_weights(shape, name, bias=True):
    target_dim = shape[-1]
    weights = tf.get_variable(name+'w', shape, initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(target_dim)))
    biases = tf.get_variable(name+'b', [target_dim], initializer=tf.constant_initializer(0.0)) if bias else None
    return weights, biases

def ff_multilayer(ini_layer, hidden_sizes, non_linear_function=tf.nn.relu, keep_prob=None, name=''):
    current_layer = ini_layer
    for i, hidden_dim in enumerate(hidden_sizes):
        current_layer = add_layer(current_layer, hidden_dim, non_linear_function=non_linear_function, keep_prob=keep_prob, name=name+'layer' + str(i))
    return current_layer

def add_linear_layer(layer, hidden_size, keep_prob=None, name=''):
    return add_layer(layer, hidden_size, non_linear_function=None, keep_prob=keep_prob, name=name)

def add_layer(layer, hidden_size, non_linear_function=tf.nn.relu, keep_prob=None, name=''):
    weight, bias = projection_weights(layer.get_shape().as_list()[1], hidden_size, name=name)
    activation = tf.matmul(layer, weight) + bias
    if non_linear_function is not None:
        activation = non_linear_function(activation)
    if keep_prob is None:
        return activation
    else:
        return tf.nn.dropout(activation, keep_prob=keep_prob)



#--------------------------------------------------------------
# Run helpers
#--------------------------------------------------------------

def restore_checkpoint(saver, session, checkpoint_dir, checkpoint_path=None):
    if not checkpoint_path:
        print('Restoring last checkpoint in %s' % checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if not (ckpt and ckpt.model_checkpoint_path):
            raise Exception('Error: checkpoint directory %s not found or accessible.' % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Restoring checkpoint %s' % os.path.join(checkpoint_dir,checkpoint_path))
        saver.restore(session, os.path.join(checkpoint_dir,checkpoint_path))

class TensorboardData:
    def __init__(self, generate_tensorboard_data=True):
        self.generate_tensorboard_data=generate_tensorboard_data

    def open(self, summaries_dir, run_name, graph):
        if self.generate_tensorboard_data:
            train_path = summaries_dir + '/train_' + run_name
            valid_path = summaries_dir + '/valid_' + run_name
            if os.path.exists(train_path): shutil.rmtree(train_path)
            if os.path.exists(valid_path): shutil.rmtree(valid_path)
            self.train_writer = tf.train.SummaryWriter(summaries_dir + '/train_' + run_name, graph, flush_secs=30)
            self.valid_writer = tf.train.SummaryWriter(summaries_dir + '/valid_' + run_name, graph, flush_secs=120)

    def add_train_summary(self, summary, step):
        if self.generate_tensorboard_data:
            self.train_writer.add_summary(summary, step)

    def add_valid_summary(self, summary, step):
        if self.generate_tensorboard_data:
            self.valid_writer.add_summary(summary, step)

    def close(self):
        if self.generate_tensorboard_data:
            self.train_writer.close()
            self.valid_writer.close()

def count_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters






