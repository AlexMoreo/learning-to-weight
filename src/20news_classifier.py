from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf
from helpers import *
from helpers import *
from pprint import pprint

#TODO: count number of parameters
#TODO: filter headers, footers, quotes, stop-words, and clean empty docs
#TODO: take weights in tfidf into account
#TODO: hashing + dimension_embeddings
#TODO: sup weighting
#TODO: clustering and two losses
class Dataset:
    def __init__(self, valid_samples=1000, categories=None, vectorize='hashing'):
        err_exit(vectorize not in ['hashing', 'tfidf', 'count'], err_msg='Param vectorize should be one in "hashing", "tfidf", or "count"')
        self.train = fetch_20newsgroups(subset='train', categories=categories)#,remove=('headers', 'footers', 'quotes'))
        self.test  = fetch_20newsgroups(subset='test', categories=categories)#, remove=('headers', 'footers', 'quotes'))
        if vectorize == 'hashing':
            vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
        elif vectorize == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english')
        else:
            vectorizer = CountVectorizer(stop_words='english')
        #vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
        self.train_vec = vectorizer.fit_transform(self.train.data)
        self.train_vec.sort_indices()
        self.test_vec = vectorizer.transform(self.test.data)
        self.test_vec.sort_indices()
        self.valid_samples=valid_samples
        self.epoch=0
        self.offset=0

    def valid_offset(self):
        return self.num_tr_docs()-self.valid_samples

    def get_index_values(self, batch):
        num_indices = len(batch.nonzero()[0])
        indices = batch.nonzero()[0].reshape(num_indices, 1)
        values = batch.nonzero()[1]
        return indices, values

    def train_batch(self, batch_size=64):
        to_pos = min(self.offset + batch_size, self.valid_offset())
        batch = self.train_vec[self.offset:to_pos]
        indices, values = self.get_index_values(batch)
        labels = self.train.target[self.offset:to_pos]
        self.offset = self.offset + batch_size
        if self.offset >= self.valid_offset():
            self.offset = 0
            self.epoch += 1
        return indices, values, labels

    def val_batch(self):
        batch = self.train_vec[self.valid_offset() : ]
        indices, values = self.get_index_values(batch)
        labels = self.train.target[self.valid_offset() : ]
        return indices, values, labels

    def test_batch(self):
        indices, values = self.get_index_values(self.test_vec)
        return indices, values, self.test.target

    def num_tr_docs(self):
        return len(self.train.filenames)

    def num_categories(self):
        return len(np.unique(self.train.target))

    def num_features(self):
        return self.train_vec.shape[1]

def main(argv=None):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    #categories = ['alt.atheism', 'talk.religion.misc']
    data = Dataset(categories=categories, vectorize='tfidf')

    x_size = data.num_features()
    n_classes = data.num_categories()
    embedding_size = 500
    hidden_sizes = [1000, 500]
    drop_k_p=0.5

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x_indices = tf.placeholder(tf.int64, shape=[None, 1])
      x_values = tf.placeholder(tf.int64, shape=[None])
      x = tf.SparseTensor(indices=x_indices, values=x_values, shape=[3])
      y = tf.placeholder(tf.int64, shape=[None])
      embeddings = tf.Variable(tf.random_uniform([x_size, embedding_size], -1.0, 1.0))
      keep_p = tf.placeholder(tf.float32, name='dropoupt_keep_p')

      def add_layer(layer, hidden_size, drop, name):
          weight, bias = projection_weights(layer.get_shape().as_list()[1], hidden_size, name=name)
          activation = tf.nn.relu(tf.matmul(layer,weight)+bias)
          if drop:
              return tf.nn.dropout(activation, keep_prob=keep_p)
          else:
              return activation

      doc_embed = tf.nn.embedding_lookup_sparse(embeddings, x, None)
      current_layer = doc_embed
      for i,hidden_dim in enumerate(hidden_sizes):
          current_layer = add_layer(current_layer, hidden_dim, drop=True, name='layer'+str(i))
      y_ = add_layer(current_layer, n_classes, drop=False, name='output_layer')

      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_,y))

      correct_prediction = tf.equal(y, tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100

      op_step = tf.Variable(0, trainable=False)
      rate = tf.train.exponential_decay(.1, op_step, 1, 0.9999)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)


    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------

    show_step = 100
    valid_step = show_step * 10
    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %.2fM' % (n_params/1E6))
        tf.initialize_all_variables().run()
        l_ave=0.0
        offset = 0
        for step in range(1,1000000):
            x_ind, x_val, y_lab = data.train_batch()
            _,l,lr = session.run([optimizer, loss, rate], feed_dict={x_indices:x_ind, x_values:x_val,y:y_lab,keep_p:drop_k_p})
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][lr=%.5f] loss=%.7f' % (step, lr, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                x_ind, x_val, y_lab = data.val_batch()
                #val_acc = tf.contrib.metrics.accuracy(predictions, y_lab)
                val_acc = accuracy.eval(feed_dict={x_indices: x_ind, x_values: x_val, y: y_lab, keep_p: 1.0})
                print('Validation accuracy %.3f%%' % val_acc)

                x_ind, x_val, y_lab = data.test_batch()
                test_acc = accuracy.eval(feed_dict={x_indices: x_ind, x_values: x_val, y: y_lab, keep_p: 1.0})
                print('Test accuracy %.3f%%' % test_acc)





#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    tf.app.run()