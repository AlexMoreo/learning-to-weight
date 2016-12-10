import numpy as np
import tensorflow as tf
from helpers import *
from helpers import *
from pprint import pprint
import time
from dataset_loader import Dataset
from sklearn.metrics import f1_score

#TODO: clustering and two losses

def main(argv=None):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data = Dataset(categories=categories, vectorize='hashing', delete_metadata=True)
    print("|Tr|=%d" % data.num_tr_documents())
    print("|Va|=%d" % data.num_val_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|C|=%d" % data.num_categories())
    print("Vectorizer=%s" % data.vectorize)

    x_size = data.num_features()
    n_classes = data.num_categories()
    embedding_size = 500
    hidden_sizes = [500,100]
    drop_k_p=0.5

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x_indices = tf.placeholder(tf.int64, shape=[None, 1])
      x_values = tf.placeholder(tf.int64, shape=[None])
      x_weights = tf.placeholder(tf.float32, shape=[None])
      x = tf.SparseTensor(indices=x_indices, values=x_values, shape=[3])
      sp_w = tf.SparseTensor(indices=x_indices, values=x_weights, shape=[3]) if data.is_weighted() else None
      y = tf.placeholder(tf.int64, shape=[None])
      #embeddings = tf.Variable(tf.random_uniform([x_size, embedding_size], -1.0, 1.0)) # with learning_rate = 0.01
      embeddings = tf.Variable(tf.truncated_normal([x_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))) # with learning_rate = 0.1

      keep_p = tf.placeholder(tf.float32, name='dropoupt_keep_p')

      doc_embed = tf.nn.embedding_lookup_sparse(params=embeddings, sp_ids=x, sp_weights=sp_w, combiner='mean') # "mean", "sqrtn", "sum"
      ff_out = ff_multilayer(doc_embed, hidden_sizes=hidden_sizes, keep_prob=keep_p, name='ff_multilayer')
      logits = add_layer(ff_out, n_classes, name='output_layer')

      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y))

      y_ = tf.nn.softmax(logits)
      prediction = tf.argmax(y_, 1)
      correct_prediction = tf.equal(y, prediction)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100
      #accuracy = tf.contrib.metrics.accuracy(y_, y)

      op_step = tf.Variable(0, trainable=False)
      rate = tf.train.exponential_decay(0.1, op_step, 1, 0.9999)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)


    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, training=False):
        (x_ind, x_val, x_wei), y_lab = batch_parts
        return {x_indices: x_ind, x_values: x_val, x_weights: x_wei, y: y_lab, keep_p: drop_k_p if training else 1.0}

    show_step = 100
    valid_step = show_step * 10
    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %.2fM' % (n_params/1E6))
        tf.initialize_all_variables().run()
        l_ave=0.0
        timeref = time.time()
        for step in range(1,1000000):
            _,l,lr = session.run([optimizer, loss, rate], feed_dict=as_feed_dict(data.train_batch(), training=True))
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][ep=%d][lr=%.5f] loss=%.7f' % (step, data.epoch, lr, l_ave / show_step))
                l_ave = 0.0


            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))

                val_acc = accuracy.eval(feed_dict=as_feed_dict(data.val_batch()))
                print('Validation accuracy %.3f%%' % val_acc)

                #test_acc = accuracy.eval(feed_dict=as_feed_dict(data.test_batch()))
                test_dict = as_feed_dict(data.test_batch())
                test_acc, predictions = session.run([accuracy, prediction], feed_dict=test_dict)
                microF1 = f1_score(test_dict[y], predictions, average='micro')
                macroF1 = f1_score(test_dict[y], predictions, average='macro')
                print('Test [accuracy=%.3f%%, MacroF1=%.3f, microF1=%.3f' % (test_acc, macroF1, microF1))

                timeref = time.time()


#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    tf.app.run()