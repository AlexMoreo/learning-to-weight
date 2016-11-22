import numpy as np
import tensorflow as tf
from helpers import *
from pprint import pprint
import time
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys
from sklearn import svm

#TODO: save weights only if f1 improves
#TODO: save weights of the best performing configuration, not the last one after early-stop
#TODO: convolution on the supervised feat-cat statistics
#TODO: convolution on the supervised feat-cat statistics + freq (L1)
#TODO: convolution on the supervised feat-cat statistics + freq (L1) + prob C (could be useful for non binary class)
def main(argv=None):

    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs

    categories = None #['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data = Dataset(categories=categories, vectorize='count', delete_metadata=True, dense=True, positive_cat=pos_cat_code, feat_sel=feat_sel)
    if data.vectorize=='count':
        print('L1-normalize')
        data.devel_vec = normalize(data.devel_vec, norm='l1', axis=1, copy=False)
        data.test_vec  = normalize(data.test_vec, norm='l1', axis=1, copy=False)
    print("|Tr|=%d" % data.num_tr_documents())
    print("|Va|=%d" % data.num_val_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_ategories())))
    print("Prevalence of positive class: %.3f" % data.class_prevalence())
    print("Vectorizer=%s" % data.vectorize)

    checkpoint_dir='.'

    print('Getting supervised correlations')
    sup = [data.feat_sup_statistics(f,cat_label=1) for f in range(data.num_features())]
    feat_corr_info = np.concatenate([[sup_i.tpr(), sup_i.fpr()] for sup_i in sup])
    #feat_corr_info = np.concatenate([[sup_i.p_tp(), sup_i.p_fp(), sup_i.p_fn(), sup_i.p_tn()] for sup_i in sup])
    info_by_feat = len(feat_corr_info) / data.num_features()

    x_size = data.num_features()
    batch_size = FLAGS.batchsize

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
          outs = FLAGS.hidden
          #filter = tf.Variable(tf.random_normal([info_by_feat, 1, outs], stddev=.1))
          filter = tf.Variable(tf.truncated_normal([info_by_feat, 1, outs], stddev=1.0 / math.sqrt(outs)))
          #filter_bias = tf.Variable(tf.zeros(outs))
          filter_bias = tf.Variable(0.1, reshape=[outs])
          conv = tf.nn.conv1d(idf_tensor, filters=filter, stride=info_by_feat, padding='VALID')
          relu = tf.nn.relu(tf.nn.bias_add(conv, filter_bias))
          reshape = tf.reshape(relu, [data.num_features(), outs])
          idf = tf.reshape(add_layer(reshape,1), [data.num_features()])
          return idf

      weighted_layer = tf.mul(tf_like(x), idf_like())
      normalized = tf.nn.l2_normalize(weighted_layer, dim=1) if FLAGS.normalize else weighted_layer
      logits = tf.squeeze(add_linear_layer(normalized, 1, name='out_logistic_function'))

      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,y))

      y_ = tf.nn.sigmoid(logits)

      prediction = tf.round(y_) # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise
      correct_prediction = tf.equal(y, prediction)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100

      if FLAGS.optimizer == 'sgd':
          op_step = tf.Variable(0, trainable=False)
          decay = tf.train.exponential_decay(FLAGS.lrate, op_step, 1, 0.9999)
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=decay).minimize(loss)
      else:#adam
          optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss) # .005

      saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts):
        x_, y_ = batch_parts
        return {x: x_, y: y_}

    best_val = dict({'acc':0.0, 'f1':0.0, 'p':0.0, 'r':0.0})
    best_test = dict({'acc': 0.0, 'f1': 0.0, 'p': 0.0, 'r': 0.0})

    def evaluation(batch, best_score):
        eval_dict = as_feed_dict(batch)
        acc, predictions = session.run([accuracy, prediction], feed_dict=eval_dict)
        f1 = f1_score(eval_dict[y], predictions, average='binary', pos_label=1)
        p = precision_score(eval_dict[y], predictions, average='binary', pos_label=1)
        r = recall_score(eval_dict[y], predictions, average='binary', pos_label=1)
        improvement = (f1 > best_score['f1'])
        if not improvement and f1 == 0.0:
            improvement = (p > best_score['p']) or (r > best_score['r'])
        best_score['acc'] = max(acc, best_score['acc'])
        best_score['f1'] = max(f1, best_score['f1'])
        best_score['p'] = max(p, best_score['p'])
        best_score['r'] = max(r, best_score['r'])
        return acc, f1, p, r, improvement

    show_step = 100
    valid_step = show_step * 10
    last_improvement = 0
    early_stop_steps = 10
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

                acc, f1, p, r, improves = evaluation(data.val_batch(), best_score=best_val)
                print('Validation acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f %s' % (acc, f1, p, r, ('[improves]' if improves else '')))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savemodel(session, step, saver, checkpoint_dir, 'model')
                elif f1 == 0.0 and last_improvement > 5:
                    print 'Reinitializing model parameters'
                    tf.initialize_all_variables().run()
                    last_improvement = 0

                acc, f1, p, r, _ = evaluation(data.test_batch(), best_score=best_test)
                print('[Test acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f]' % (acc, f1, p, r))

                timeref = time.time()

            #early stop if not improves after 10 validations
            if last_improvement >= early_stop_steps:
                print('Early stop after %d validation steps without improvements' % last_improvement)
                break

        fout_path = 'out-'+FLAGS.optimizer+'-'+str(FLAGS.lrate)+('-norm' if FLAGS.normalize else '')+'-c'+str(pos_cat_code)+'.txt'
        with open(fout_path, 'w') as fout:
            def tee(outstring, fout):
                print outstring
                fout.write(outstring+'\n')

            print 'Test evaluation:'
            restore_checkpoint(saver, session, checkpoint_dir)
            acc, f1, p, r, _ = evaluation(data.test_batch(), best_score=best_test)
            tee('Logistic Regression acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, p, r), fout)

            print 'Weighting documents'
            devel_x, devel_y = data.get_devel_set()
            devel_x_weighted = normalized.eval(feed_dict={x:devel_x})
            test_x, test_y   = data.test_batch()
            test_x_weighted  = normalized.eval(feed_dict={x:test_x})

            C = 1.0  # SVM regularization parameter
            #svc = svm.SVC(kernel='linear', C=C).fit(devel_x_weighted, devel_y)
            #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(devel_x_weighted, devel_y)
            #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(devel_x_weighted, devel_y)
            lin_svc = svm.LinearSVC(C=C).fit(devel_x_weighted, devel_y)

            print 'Getting predictions'
            def evaluation(classifier, test, true_labels):
                predictions = classifier.predict(test)
                acc = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)
                p = precision_score(true_labels, predictions, average='binary', pos_label=1)
                r = recall_score(true_labels, predictions, average='binary', pos_label=1)
                return acc, f1, p, r

            #evaluation(svc, test_x_weighted, test_y)
            #evaluation(rbf_svc, test_x_weighted, test_y)
            #evaluation(poly_svc, test_x_weighted, test_y)
            acc, f1, p, r = evaluation(lin_svc, test_x_weighted, test_y)
            tee('LinearSVM acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc * 100, f1, p, r), fout)


#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('fs', None, 'Indicates the number of features to be selected (default None --all).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 32, 'Size of the batches (default 32).')
    flags.DEFINE_integer('hidden', 100, 'Number of hidden nodes (default 100).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)')
    flags.DEFINE_string('optimizer', 'sgd', 'Optimization algorithm in ["sgd", "adam"] (default sgd)')
    flags.DEFINE_boolean('normalize', True, 'Imposes normalization to the document vectors (default True)')
    #flags.DEFINE_string('fout', '', 'Output file')

    err_exit(FLAGS.optimizer not in ['sgd','adam'],err_msg="Param error: optimizer should be either 'sgd' or 'adam'")

    tf.app.run()