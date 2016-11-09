from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from helpers import *
from helpers import *
from pprint import pprint
import time

#TODO: sup weighting
#TODO: clustering and two losses
class Dataset:
    def __init__(self, valid_samples=1000, categories=None, vectorize='hashing'):
        err_exit(vectorize not in ['hashing', 'tfidf', 'count'], err_msg='Param vectorize should be one in "hashing", "tfidf", or "count"')
        self.train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
        self.test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers', 'footers', 'quotes'))
        self.send_weights=False
        if vectorize == 'hashing':
            vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
        elif vectorize == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english')
            self.send_weights = True
        else:
            vectorizer = CountVectorizer(stop_words='english')
        self.train_vec = vectorizer.fit_transform(self.train.data)
        self.train_vec.sort_indices()
        self.test_vec = vectorizer.transform(self.test.data)
        self.test_vec.sort_indices()
        self.valid_samples=valid_samples
        self.epoch=0
        self.offset=0
        self.train_indexes=self.get_doc_indexes(self.train_vec[:self.valid_offset()])
        self.valid_indexes = self.get_doc_indexes(self.train_vec[self.valid_offset():], offset=self.valid_offset())
        self.test_indexes = self.get_doc_indexes(self.test_vec)

    #virtually remove invalid documents (documents without any feature)
    def get_doc_indexes(self, vector_set, offset=0):
        all_indexes = np.arange(offset, offset + vector_set.shape[0])
        return [i for i in all_indexes if vector_set[i-offset].nnz>0]

    def valid_offset(self):
        return self.num_devel_docs()-self.valid_samples

    def get_index_values(self, batch):
        num_indices = len(batch.nonzero()[0])
        indices = batch.nonzero()[0].reshape(num_indices, 1)
        values = batch.nonzero()[1]
        return indices, values

    def get_weights(self, batch):
        return batch.data if self.send_weights else []

    def train_batch(self, batch_size=64):
        to_pos = min(self.offset + batch_size, self.num_tr_documents())
        batch = self.train_vec[self.train_indexes[self.offset:to_pos]]
        indices, values = self.get_index_values(batch)
        weights = self.get_weights(batch)
        labels = self.train.target[self.train_indexes[self.offset:to_pos]]
        self.offset = self.offset + batch_size
        if self.offset >= self.num_tr_documents():
            self.offset = 0
            self.epoch += 1
            random.shuffle(self.train_indexes)
        return indices, values, weights, labels

    def val_batch(self):
        batch = self.train_vec[self.valid_indexes]
        indices, values = self.get_index_values(batch)
        weights = self.get_weights(batch)
        labels = self.train.target[self.valid_indexes]
        return indices, values, weights, labels

    def test_batch(self):
        indices, values = self.get_index_values(self.test_vec[self.test_indexes])
        weights = self.get_weights(self.test_vec)
        return indices, values, weights, self.test.target[self.test_indexes]

    def num_devel_docs(self):
        return len(self.train.filenames)

    def num_categories(self):
        return len(np.unique(self.train.target))

    def num_features(self):
        return self.train_vec.shape[1]

    def num_tr_documents(self):
        return len(self.train_indexes)

    def num_val_documents(self):
        return len(self.valid_indexes)

    def num_test_documents(self):
        return len(self.test_indexes)

def main(argv=None):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    #categories = ['alt.atheism', 'talk.religion.misc']
    data = Dataset(categories=categories, vectorize='hashing')
    print("|Tr|=%d" % data.num_tr_documents())
    print("|Va|=%d" % data.num_val_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|C|=%d" % data.num_categories())

    x_size = data.num_features()
    n_classes = data.num_categories()
    embedding_size = 500
    hidden_sizes = [500,50]
    drop_k_p=0.5

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x_indices = tf.placeholder(tf.int64, shape=[None, 1])
      x_values = tf.placeholder(tf.int64, shape=[None])
      x_weights = tf.placeholder(tf.float32, shape=[None])
      x = tf.SparseTensor(indices=x_indices, values=x_values, shape=[3])
      sp_w = tf.SparseTensor(indices=x_indices, values=x_weights, shape=[3]) if data.send_weights else None
      y = tf.placeholder(tf.int64, shape=[None])
      embeddings = tf.Variable(tf.random_uniform([x_size, embedding_size], -1.0, 1.0))
      keep_p = tf.placeholder(tf.float32, name='dropoupt_keep_p')

      doc_embed = tf.nn.embedding_lookup_sparse(params=embeddings, sp_ids=x, sp_weights=sp_w)
      ff_out = ff_multilayer(doc_embed, hidden_sizes=hidden_sizes, keep_prob=keep_p, name='ff_multilayer')
      y_ = add_layer(ff_out, n_classes, name='output_layer')

      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_,y))

      correct_prediction = tf.equal(y, tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100
      #accuracy = tf.contrib.metrics.accuracy(y_, y)

      op_step = tf.Variable(0, trainable=False)
      rate = tf.train.exponential_decay(0.01, op_step, 1, 0.9999)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)


    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, training=False):
        x_ind, x_val, x_wei, y_lab = batch_parts
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

                test_acc = accuracy.eval(feed_dict=as_feed_dict(data.test_batch()))
                print('Test accuracy %.3f%%' % test_acc)

                timeref = time.time()


#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    tf.app.run()