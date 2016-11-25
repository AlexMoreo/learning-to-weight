import numpy as np
import tensorflow as tf
from helpers import *
from pprint import pprint
import time
from feature_selection_function import *
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys
from baseline_classification import train_classifiers
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#TODO: balanced batchs
#TODO: tensorboard
#TODO: improve result out
#TODO: select FS function, and plot to file
#TODO: convolution on the supervised feat-cat statistics + freq (L1)
#TODO: convolution on the supervised feat-cat statistics + freq (L1) + prob C (could be useful for non binary class)
def main(argv=None):
    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs
    categories = None if not FLAGS.debug else ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
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

    print('Getting supervised correlations')
    sup = [data.feat_sup_statistics(f,cat_label=1) for f in range(data.num_features())]
    feat_corr_info = np.concatenate([[sup_i.tpr(), sup_i.fpr()] for sup_i in sup])
    #feat_corr_info = np.concatenate([[sup_i.p_tp(), sup_i.p_fp(), sup_i.p_fn(), sup_i.p_tn()] for sup_i in sup])
    info_by_feat = len(feat_corr_info) / data.num_features()

    x_size = data.num_features()
    batch_size = FLAGS.batchsize
    drop_keep_p = 0.8

    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None])
        keep_p = tf.placeholder(tf.float32)

        feat_info = tf.constant(feat_corr_info, dtype=tf.float32)

        def tf_like(x_raw):
            tf_param = tf.Variable(tf.ones([1]), tf.float32)
            return tf.mul(tf.log(x_raw + tf.ones_like(x_raw)), tf_param)
            #return tf.log(x_raw + tf.ones_like(x_raw))
            #x_stack = tf.reshape(x_raw, shape=[-1,1])
            #tfx_stack = tf.squeeze(ff_multilayer(x_stack,[1],non_linear_function=tf.nn.sigmoid))
            #return tf.reshape(tfx_stack, shape=[-1, x_size])

        def idf_like(info_arr):
            #return tf.ones(shape=[data.num_features()], dtype=tf.float32)
            #filter = tf.Variable(tf.truncated_normal([info_by_feat, 1, FLAGS.hidden], stddev=1.0 / math.sqrt(FLAGS.hidden)))
            filter = tf.Variable(tf.random_normal([info_by_feat, 1, FLAGS.hidden], stddev=.01 / math.sqrt(FLAGS.hidden)))
            #filter = tf.Variable(tf.random_normal([info_by_feat, 1, FLAGS.hidden], stddev=.00001))
            filter_bias = tf.Variable(tf.zeros(FLAGS.hidden))
            #filter_bias = tf.Variable(tf.random_uniform([FLAGS.hidden], 0.01, 1.0 / math.sqrt(FLAGS.hidden)))

            print info_arr.get_shape()
            n_results = info_arr.get_shape().as_list()[-1] / info_by_feat
            print n_results
            idf_tensor = tf.reshape(info_arr, shape=[1, -1, 1])
            conv = tf.nn.conv1d(idf_tensor, filters=filter, stride=info_by_feat, padding='VALID')
            relu = tf.nn.relu(tf.nn.bias_add(conv, filter_bias))
            relu = tf.nn.dropout(relu, keep_prob=keep_p)
            reshape = tf.reshape(relu, [n_results, FLAGS.hidden])
            idf = tf.reshape(add_layer(reshape,1), [n_results])
            print idf.get_shape()
            return idf

        weighted_layer = tf.mul(tf_like(x), idf_like(feat_info))
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
        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)
        else:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lrate).minimize(loss)  # 0.0001

        #pre-learn the idf-like function as any feature selection function
        x_func = tf.placeholder(tf.float32, shape=[None, info_by_feat])
        y_func = tf.placeholder(tf.float32, shape=[None])
        idf_prediction = idf_like(x_func)
        idf_loss = tf.reduce_mean(tf.square(tf.sub(y_func, idf_prediction)))
        idf_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(idf_loss) #0.0001

        saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, dropout=False):
        x_, y_ = batch_parts
        return {x: x_, y: y_, keep_p: drop_keep_p if dropout else 1.0}

    best_val = dict({'acc':0.0, 'f1':0.0, 'p':0.0, 'r':0.0})
    best_test = dict({'acc': 0.0, 'f1': 0.0, 'p': 0.0, 'r': 0.0})

    def evaluation(batch, best_score):
        eval_dict = as_feed_dict(batch, dropout=False)
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

    checkpoint_dir = FLAGS.checkpointdir
    create_if_not_exists(checkpoint_dir)
    pc = data.class_prevalence()

    def supervised_idf(tpr, fpr):
        return information_gain(tpr, fpr, pc)
        #return chi_square(tpr, fpr, pc)
        #return gss(tpr, fpr, pc)

    def sample():
        x = [random.random() for _ in range(info_by_feat)]
        y = supervised_idf(tpr=x[0], fpr=x[1])
        return (x, y)

    def pretrain_batch(batch_size=1):
        next = [sample() for _ in range(batch_size)]
        return zip(*next)

    def plot_coordinates(div=50):
        x1_div = np.linspace(0.0, 1.0, div)
        x2_div = np.linspace(0.0, 1.0, div)
        points = itertools.product(x1_div, x2_div)
        return zip(*[p for p in points])

    def comp_plot(x1, x2, y_):
        y = [supervised_idf(x1[i], x2[i]) for i in range(len(x1))]
        fig = plt.figure(figsize=plt.figaspect(0.4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title('Target function')
        ax.plot_trisurf(x1, x2, y, linewidth=0.2, cmap=cm.jet)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title('Learnt function')
        ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet)
        plt.show()

    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()

        #pre-train the idf-like parameters
        if FLAGS.pretrain:
            l_ave = 0.0
            show_step = 1000
            for step in range(1, 40001):
                x_, y_ = pretrain_batch()
                _, l = session.run([idf_optimizer, idf_loss], feed_dict={x_func: x_, y_func: y_, keep_p:drop_keep_p})
                l_ave += l

                if step % show_step == 0:
                    l_ave /= show_step
                    print('[step=%d] idf-loss=%.7f' % (step, l_ave))
                    if l_ave < 0.0003:
                        print 'Error < 0.0003, proceed'
                        break
                    l_ave = 0.0

                if FLAGS.plot and step % (show_step*10) == 0:
                    x1_, x2_ = plot_coordinates(div=40)
                    y_ = []
                    x_ = zip(x1_, x2_)
                    for xi_ in x_:
                        y_.append(idf_prediction.eval(feed_dict={x_func: [xi_], keep_p:1.0}))
                    y_ = np.reshape(y_, len(x1_))
                    comp_plot(x1_, x2_, y_)

        show_step = 100
        valid_step = show_step * 10
        last_improvement = 0
        early_stop_steps = 10
        l_ave=0.0
        timeref = time.time()
        for step in range(1,50000):
            _,l = session.run([optimizer, loss], feed_dict=as_feed_dict(data.train_batch(batch_size), dropout=True))
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

            print 'Test evaluation:'
            restore_checkpoint(saver, session, checkpoint_dir)
            acc, f1, p, r, _ = evaluation(data.test_batch(), best_score=best_test)
            tee('Logistic Regression acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, p, r), fout)

            print 'Weighting documents'
            devel_x, devel_y = data.get_devel_set()
            devel_x_weighted = normalized.eval(feed_dict={x:devel_x})
            test_x, test_y   = data.test_batch()
            test_x_weighted  = normalized.eval(feed_dict={x:test_x})

            train_classifiers(devel_x_weighted, devel_y, test_x_weighted, test_y, fout)

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('fs', None, 'Indicates the number of features to be selected (default None --all).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 32, 'Size of the batches (default 32).')
    flags.DEFINE_integer('hidden', 100, 'Number of hidden nodes (default 100).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)') #3e-4
    flags.DEFINE_string('optimizer', 'sgd', 'Optimization algorithm in ["sgd", "adam"] (default sgd)')
    flags.DEFINE_boolean('normalize', True, 'Imposes normalization to the document vectors (default True)')
    flags.DEFINE_string('checkpointdir', './model', 'Directory where to save the checkpoints of the model parameters (default "./model")')
    flags.DEFINE_boolean('pretrain', False, 'Pretrains the model parameters to mimic a given FS function (default False)')
    flags.DEFINE_boolean('debug', False, 'Set to true for fast data load, and debugging')
    flags.DEFINE_boolean('plot', False, 'Plots the idf-like function learnt')
    #flags.DEFINE_string('fout', '', 'Output file')

    err_exit(FLAGS.optimizer not in ['sgd','adam'],err_msg="Param error: optimizer should be either 'sgd' or 'adam'")
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    tf.app.run()
