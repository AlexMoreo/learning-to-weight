import numpy as np
import tensorflow as tf
from helpers import *
from pprint import pprint
import time
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys

#TODO: convolution on the supervised feat-cat statistics
#TODO: convolution on the supervised feat-cat statistics + freq (L1)
#TODO: convolution on the supervised feat-cat statistics + freq (L1) + prob C (could be useful for non binary class)
def main(argv=None):

    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data = Dataset(categories=categories, vectorize='count', delete_metadata=True, dense=True, positive_cat='talk.religion.misc')
    if data.vectorize=='count':
        print('L1-normalize')
        normalize(data.devel_vec, norm='l1', axis=1, copy=False)
        normalize(data.test_vec, norm='l1', axis=1, copy=False)
    print("|Tr|=%d" % data.num_tr_documents())
    print("|Va|=%d" % data.num_val_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_ategories())))
    print("Prevalence of positive class: %.3f" % data.class_prevalence())
    print("Vectorizer=%s" % data.vectorize)

    print('Getting supervised correlations')
    sup = [data.feat_sup_statistics(f,cat_label=1) for f in range(data.num_features())]
    feat_corr_info = np.concatenate([[sup_i.tpr(), sup_i.fpr()] for sup_i in sup])
    #feat_corr_info = np.concatenate([[sup_i.p_tp(), sup_i.p_fp(), sup_i.p_fn(), sup_i.p_tn()] for sup_i in sup])
    info_by_feat = len(feat_corr_info) / data.num_features()
    print('Creating the graph')

    x_size = data.num_features()
    batch_size = 32

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x = tf.placeholder(tf.float32, shape=[None, x_size])
      y = tf.placeholder(tf.float32, shape=[None])


      def tf_like(x_raw):
          tf_param = tf.Variable(tf.ones([1]), tf.float32)
          return tf.mul(tf.log(x_raw + tf.ones_like(x_raw)), tf_param)
          #return tf.log(x_raw + tf.ones_like(x_raw))
          #x_stack = tf.reshape(x_raw, shape=[-1,1])
          #tfx_stack = tf.squeeze(ff_multilayer(x_stack,[1],non_linear_function=tf.nn.sigmoid))
          #return tf.reshape(tfx_stack, shape=[-1, x_size])

      def idf_like():
          #return tf.ones(shape=[data.num_features()], dtype=tf.float32)
          feat_info = tf.constant(feat_corr_info, dtype=tf.float32)
          idf_tensor = tf.reshape(feat_info, shape=[1, -1, 1])
          outs = 100
          filter = tf.Variable(tf.random_normal([info_by_feat, 1, outs]))
          filter_bias = tf.Variable(tf.zeros(outs))
          conv = tf.nn.conv1d(idf_tensor, filters=filter, stride=info_by_feat, padding='VALID')
          relu = tf.nn.relu(tf.nn.bias_add(conv, filter_bias))
          reshape = tf.reshape(relu, [data.num_features(), outs])
          idf = tf.reshape(add_layer(reshape,1), [data.num_features()])
          return idf

      weighted_layer = tf.mul(tf_like(x), idf_like())
      logits = tf.squeeze(add_linear_layer(weighted_layer, 1, name='out_logistic_function'))

      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,y))

      y_ = tf.nn.sigmoid(logits)

      prediction = tf.round(y_) # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise
      correct_prediction = tf.equal(y, prediction)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100
      #accuracy = tf.contrib.metrics.accuracy(y_, y)

      #op_step = tf.Variable(0, trainable=False)
      #rate = tf.train.exponential_decay(.01, op_step, 1, 0.99999)
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)
      optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss) # 0.001

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts):
        x_, y_ = batch_parts
        return {x: x_, y: y_}

    show_step = 1
    valid_step = show_step * 10
    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()
        l_ave=0.0
        timeref = time.time()
        for step in range(1,1000000):
            _,l = session.run([optimizer, loss], feed_dict=as_feed_dict(data.train_batch(batch_size)))
            l_ave += l

            if step % show_step == 0:
                #print('[step=%d][ep=%d][lr=%.5f] loss=%.7f' % (step, data.epoch, lr, l_ave / show_step))
                print('[step=%d][ep=%d] loss=%.10f' % (step, data.epoch, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))

                val_acc = accuracy.eval(feed_dict=as_feed_dict(data.val_batch()))
                print('Validation accuracy %.3f%%' % val_acc)

                test_dict = as_feed_dict(data.test_batch())
                test_acc, predictions = session.run([accuracy, prediction], feed_dict=test_dict)
                f1 = f1_score(test_dict[y], predictions, average='binary', pos_label=1)
                p  = precision_score(test_dict[y], predictions, average='binary', pos_label=1)
                r  = recall_score(test_dict[y], predictions, average='binary', pos_label=1)
                print('Test [accuracy=%.3f%%, f1-score=%.3f, p=%.3f, r=%.3f' % (test_acc, f1, p, r))

                timeref = time.time()

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    tf.app.run()