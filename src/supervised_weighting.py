import numpy as np
import numpy as np
import tensorflow as tf
from helpers import *
from pprint import pprint
import time
from time import gmtime, strftime
import feature_selection_function
from dataset_loader import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys
from weighted_vectors import WeightedVectors
from classification_benchmark import *

#TODO: add information to sup_info, e.g., idf, ig, ptp ptn pfp pfn
#TODO: check without sigmoid, and with relu
#TODO: ConfWeight
#TODO: check out the separability index
#TODO: add a new non-linear classifier
#TODO: parallelize supervised info vector
#TODO: convolution on the supervised feat-cat statistics + freq (L1) + prob C (could be useful for non binary class)
def main(argv=None):
    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    init_time = time.time()
    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs
    data = DatasetLoader(dataset=FLAGS.dataset, vectorize='count', rep_mode='dense', positive_cat=pos_cat_code, feat_sel=feat_sel)
    print('L1-normalize')
    data.devel_vec = normalize(data.devel_vec, norm='l1', axis=1, copy=False)
    data.test_vec  = normalize(data.test_vec, norm='l1', axis=1, copy=False)
    print("|Tr|=%d [prev+ %f]" % (data.num_tr_documents(), data.train_class_prevalence()))
    print("|Val|=%d [prev+ %f]" % (data.num_val_documents(), data.valid_class_prevalence()))
    print("|Te|=%d [prev+ %f]" % (data.num_test_documents(), data.test_class_prevalence()))
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_categories())))

    print('Getting supervised correlations')
    sup = [data.feature_label_contingency_table(f, cat_label=1) for f in range(data.num_features())]
    feat_corr_info = [[sup_i.tpr(), sup_i.fpr()] for sup_i in sup]
    #feat_corr_info = np.concatenate([[sup_i.p_tp(), sup_i.p_fp(), sup_i.p_fn(), sup_i.p_tn()] for sup_i in sup])
    info_by_feat = len(feat_corr_info[0])

    x_size = data.num_features()
    batch_size = FLAGS.batchsize if FLAGS.batchsize!=-1 else data.num_tr_documents()
    drop_keep_p = 0.8

    # graph definition --------------------------------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None])
        keep_p = tf.placeholder(tf.float32)

        feat_info = tf.constant(np.concatenate(feat_corr_info), dtype=tf.float32)

        def tf_like(x_raw):
            tf_param = tf.Variable(tf.ones([1]), tf.float32)
            return tf.mul(tf.log(x_raw + tf.ones_like(x_raw)), tf_param)
            #return tf.log(x_raw + tf.ones_like(x_raw))
            #x_stack = tf.reshape(x_raw, shape=[-1,1])
            #tfx_stack = tf.squeeze(ff_multilayer(x_stack,[1],non_linear_function=tf.nn.sigmoid))
            #return tf.reshape(tfx_stack, shape=[-1, x_size])

        def idf_like(info_arr):
            #return tf.ones(shape=[data.num_features()], dtype=tf.float32)
            filter_weights = tf.get_variable('idf_weights', [info_by_feat, 1, FLAGS.hidden], initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(FLAGS.hidden)))
            filter_biases  = tf.get_variable('idf_biases', [FLAGS.hidden], initializer=tf.constant_initializer(0.0))
            proj_weights   = tf.get_variable('proj_weights', [FLAGS.hidden, 1], initializer=tf.random_normal_initializer(stddev=1.))
            proj_biases    = tf.get_variable('proj_bias', [1], initializer=tf.constant_initializer(0.0))

            #activation=tf.nn.relu
            n_results = info_arr.get_shape().as_list()[-1] / info_by_feat
            idf_tensor = tf.reshape(info_arr, shape=[1, -1, 1])
            conv = tf.nn.conv1d(idf_tensor, filters=filter_weights, stride=info_by_feat, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [n_results, FLAGS.hidden])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            if FLAGS.forcepos:
                if FLAGS.linidf:
                    proj = tf.nn.relu(proj)
                else:
                    proj = tf.nn.sigmoid(proj)
            else:
                if not FLAGS.linidf:
                    proj = tf.nn.tanh(proj)
            return tf.reshape(proj, [n_results])

        def full_tfidf(x_raw, info_arr):
            filter_weights = tf.get_variable('full_weights', [1+info_by_feat, 1, FLAGS.hidden], initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(FLAGS.hidden)))
            filter_biases = tf.get_variable('full_biases', [FLAGS.hidden], initializer=tf.constant_initializer(0.0))
            proj_weights = tf.get_variable('proj_weights', [FLAGS.hidden, 1], initializer=tf.random_normal_initializer(stddev=1.))
            proj_biases = tf.get_variable('proj_bias', [1], initializer=tf.constant_initializer(0.0))

            info_arr_exp = tf.expand_dims(info_arr, 0)  # from shape [info_by_feat] to shape [1, info_by_feat]
            n_rows = tf.shape(x_raw)[0]
            info_tiled = tf.tile(info_arr_exp, tf.pack([n_rows, 1]))
            feat_tf = tf.reshape(x_raw, [-1,1])
            feat_idf= tf.reshape(info_tiled, [-1,info_by_feat])
            feat_tfidf = tf.concat(1, [feat_tf, feat_idf])
            x_tfidf = tf.reshape(feat_tfidf, [n_rows,x_size*(info_by_feat+1), 1]) #shape=[rows, cols=(infobyfeat+1)xfeat, channels=1]
            conv = tf.nn.conv1d(x_tfidf, filters=filter_weights, stride=1+info_by_feat, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [-1, FLAGS.hidden])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            if FLAGS.forcepos:
                if FLAGS.linidf:
                    proj = tf.nn.relu(proj)
                else:
                    proj = tf.nn.sigmoid(proj)
            else:
                if not FLAGS.linidf:
                    proj = tf.nn.tanh(proj)
            return tf.reshape(proj, [n_rows,-1])

            #ff=ff_multilayer(feat_tfidf,[1000,50],keep_prob=keep_p)
            #full_tfidf = add_linear_layer(ff, 1)
            #relu = tf.nn.dropout(tf.nn.relu(tf.matmul(feat_tfidf,h1_weights)+h1_biases), keep_prob=keep_p)
            #full_tfidf = tf.nn.relu(tf.matmul(relu, proj_weights) + proj_biases)
            #x_weighted = tf.reshape(full_tfidf, [n_rows, x_size])
            #print 'done'
            #return x_weighted

        if FLAGS.computation == 'tfidflike':
            weighted_layer = tf.mul(tf_like(x), idf_like(feat_info))
        elif FLAGS.computation == 'full':
            weighted_layer = tf.mul(tf_like(x), full_tfidf(x, feat_info))

        print weighted_layer.get_shape()
        normalized = tf.nn.l2_normalize(weighted_layer, dim=1) if FLAGS.normalize else weighted_layer
        log_weights = tf.get_variable('log_weights', [data.num_features(), 1], initializer=tf.random_normal_initializer(stddev=.1))
        log_bias = tf.get_variable('log_biases', [1], initializer=tf.constant_initializer(0.0))
        logits = tf.squeeze(tf.nn.bias_add(tf.matmul(normalized, log_weights), log_bias))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y))

        y_ = tf.nn.sigmoid(logits)
        prediction = tf.squeeze(tf.round(y_))  # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise

        logistic_params = [log_weights, log_bias]
        if FLAGS.optimizer == 'sgd':
            op_step = tf.Variable(0, trainable=False)
            decay   = tf.train.exponential_decay(FLAGS.lrate, op_step, 1, 0.9999)
            end2end_optimizer  = tf.train.GradientDescentOptimizer(learning_rate=decay).minimize(loss)
            logistic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=decay).minimize(loss, var_list=logistic_params)
        elif FLAGS.optimizer == 'adam':
            end2end_optimizer  = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)
            logistic_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss, var_list=logistic_params)
        else:  #rmsprop
            end2end_optimizer  = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lrate).minimize(loss)  # 0.0001
            logistic_optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lrate).minimize(loss, var_list=logistic_params)

        #pre-learn the idf-like function as any feature selection function
        #if FLAGS.pretrain != 'off':
        x_func = tf.placeholder(tf.float32, shape=[None, info_by_feat])
        y_func = tf.placeholder(tf.float32, shape=[None])
        tf.get_variable_scope().reuse_variables()
        if FLAGS.computation == 'tfidflike':
            idf_prediction = idf_like(x_func)
            idf_loss = tf.reduce_mean(tf.square(tf.sub(y_func, idf_prediction)))
            idf_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(idf_loss)

        #idfloss_summary = tf.scalar_summary('loss/idf_loss', idf_loss)
        loss_summary = tf.scalar_summary('loss/loss', loss)
        log_weight_sum = variable_summaries(log_weights, 'logistic/weight', 'weights')
        log_bias_sum = variable_summaries(log_bias, 'logistic/bias', 'bias')
        #idf_weight_sum = variable_summaries(tf.get_variable('idf_weights'), 'idf/weight', 'weights')
        #idf_bias_sum = variable_summaries(tf.get_variable('idf_biases'), 'idf/bias', 'bias')
        #idf_projweight_sum = variable_summaries(tf.get_variable('proj_weights'), 'idf/proj/weight', 'projweight')
        #idf_projbias_sum = variable_summaries(tf.get_variable('proj_bias'), 'idf/proj/bias', 'projbias')
        #idf_summaries = tf.merge_summary([idfloss_summary, idf_weight_sum, idf_bias_sum, idf_projweight_sum, idf_projbias_sum])
        log_summaries = tf.merge_summary([loss_summary, log_weight_sum, log_bias_sum])
        end2end_summaries = tf.merge_summary([loss_summary, log_weight_sum, log_bias_sum,
                                      #idf_weight_sum, idf_bias_sum,
                                            #idf_projweight_sum, idf_projbias_sum
                                             ])

        saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, dropout=False):
        x_, y_ = batch_parts
        return {x: x_, y: y_, keep_p: drop_keep_p if dropout else 1.0}

    create_if_not_exists(FLAGS.checkpointdir)
    create_if_not_exists(FLAGS.summariesdir)
    create_if_not_exists(FLAGS.outdir)
    pc = data.devel_class_prevalence()

    def supervised_idf(tpr, fpr):
        if FLAGS.pretrain == 'off': return 0.0
        fsmethod = getattr(feature_selection_function, FLAGS.pretrain)
        return fsmethod(tpr, fpr, pc)

    def pretrain_batch(batch_size=1):
        def sample():
            x = [random.random() for _ in range(info_by_feat)]
            y = supervised_idf(tpr=x[0], fpr=x[1])
            return (x, y)
        next = [sample() for _ in range(batch_size)]
        return zip(*next)

    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()
        tensorboard = TensorboardData()
        tensorboard.open(FLAGS.summariesdir, 'sup_weight', session.graph)

        # pre-train -------------------------------------------------
        idf_steps = 0
        epsilon = 0.0003
        def idf_wrapper(x):
            return idf_prediction.eval(feed_dict={x_func: [x], keep_p: 1.0})
        plot = PlotIdf(FLAGS.plotmode, FLAGS.plotdir,
                       supervised_idf if FLAGS.pretrain!='off' else None, idf_wrapper, plotpoints=feat_corr_info)
        plotsteps = 100
        if FLAGS.pretrain != 'off':
            if FLAGS.plotmode in ['img', 'show']: plot.plot(step=0)
            l_ave = 0.0
            show_step = 1000
            for step in range(1, 40001):
                x_, y_ = pretrain_batch()
                _, l = session.run([idf_optimizer, idf_loss], feed_dict={x_func: x_, y_func: y_, keep_p:drop_keep_p})
                l_ave += l
                idf_steps += 1

                if step % show_step == 0:
                    #sum = idf_summaries.eval({x_func: x_, y_func: y_, keep_p:drop_keep_p})
                    #tensorboard.add_train_summary(sum, step)
                    l_ave /= show_step
                    print('[step=%d] idf-loss=%.7f' % (step, l_ave))
                    if l_ave < epsilon:
                        print 'Error < '+str(epsilon)+', proceed'
                        break
                    l_ave = 0.0

                if FLAGS.plotmode == 'vid' and step % plotsteps == 0: plot.plot(step=step)

            if FLAGS.plotmode in ['img', 'show']: plot.plot(step=step)

        # train -------------------------------------------------
        show_step = plotsteps = 10
        valid_step = show_step * 10
        last_improvement = 0
        early_stop_steps = 20
        l_ave=0.0
        timeref = time.time()
        logistic_optimization_phase = 10000
        best_f1 = 0.0
        log_steps = 0
        savedstep = -1
        for step in range(1,FLAGS.maxsteps):
            in_logistic_phase = FLAGS.pretrain!='off' and step < logistic_optimization_phase
            optimizer_ = logistic_optimizer if in_logistic_phase else end2end_optimizer
            tr_dict = as_feed_dict(data.train_batch(batch_size), dropout=True)
            _, l = session.run([optimizer_, loss], feed_dict=tr_dict)
            l_ave += l
            log_steps += 1

            if step % show_step == 0:
                sum = end2end_summaries.eval(feed_dict=tr_dict)
                tensorboard.add_train_summary(sum, step+idf_steps)
                tr_phase = 'logistic' if in_logistic_phase else 'end2end'
                print('[step=%d][ep=%d][op=%s] loss=%.10f' % (step, data.epoch, tr_phase, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                eval_dict = as_feed_dict(data.val_batch(), dropout=False)
                predictions, sum = session.run([prediction, loss_summary], feed_dict=eval_dict)
                tensorboard.add_valid_summary(sum, step+idf_steps)
                acc, f1, p, r = evaluation_metrics(predictions, eval_dict[y])
                improves = f1 > best_f1
                best_f1 = max(best_f1, f1)

                print('Validation acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f %s' % (acc, f1, p, r, ('[improves]' if improves else '')))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savedstep=step+idf_steps
                    savemodel(session, savedstep, saver, FLAGS.checkpointdir, 'model')
                #elif f1 == 0.0 and last_improvement > 5:
                    #    print 'Reinitializing model parameters'
                    #tf.initialize_all_variables().run()
                    #last_improvement = 0

                if FLAGS.computation!='full':
                    eval_dict = as_feed_dict(data.test_batch(), dropout=False)
                    predictions = prediction.eval(feed_dict=eval_dict)
                    acc, f1, p, r = evaluation_metrics(predictions, eval_dict[y])
                    print('[Test acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f]' % (acc, f1, p, r))
                timeref = time.time()

            if FLAGS.plotmode=='vid' and step % plotsteps == 0:
                plot.plot(step=step+idf_steps)

            #early stop if not improves after 10 validations
            if last_improvement >= early_stop_steps:
                print('Early stop after %d validation steps without improvements' % last_improvement)
                break
            if best_f1==1.0:
                print('Max validation score reached. End of training.')
                break

        tensorboard.close()

        # output -------------------------------------------------
        print 'Test evaluation:'
        if savedstep>0:
            restore_checkpoint(saver, session, FLAGS.checkpointdir)
        if FLAGS.plotmode in ['img', 'show']: plot.plot(step=savedstep)
        eval_dict = as_feed_dict(data.test_batch(), dropout=False)
        predictions = prediction.eval(feed_dict=eval_dict)
        acc, f1, p, r = evaluation_metrics(predictions, eval_dict[y])
        print('Logistic Regression acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, p, r))

        run_params_dic = {'num_features': data.num_features(),
                          'date': strftime("%d-%m-%Y", gmtime()),
                          'hiddensize': FLAGS.hidden,
                          'lrate': FLAGS.lrate,
                          'optimizer': FLAGS.optimizer,
                          'normalize': FLAGS.normalize,
                          'nonnegative': FLAGS.forcepos,
                          'pretrain': FLAGS.pretrain,
                          'iterations': idf_steps + log_steps,
                          'notes': FLAGS.notes + ('(linidf=%s)' % FLAGS.linidf) + ('(val_f1%.4f)' % best_f1),
                          'run': FLAGS.run}

        # if indicated, saves the result of the current logistic regressor
        if FLAGS.resultcontainer:
            results = ReusltTable(FLAGS.resultcontainer)
            data.vectorize='learned'
            results.init_row_result('LogisticRegression-Internal', data, run=FLAGS.run)
            results.add_result_metric_scores(acc=acc, f1=f1, prec=p, rec=r,
                                             cont_table=contingency_table(predictions, eval_dict[y]), init_time=init_time)
            results.set_all(run_params_dic)
            results.commit()

        print 'Weighting documents'
        train_x, train_y = data.get_train_set()
        train_x_weighted = normalized.eval(feed_dict={x: train_x, keep_p: 1.0})
        val_x, val_y   = data.get_validation_set()
        val_x_weighted = normalized.eval(feed_dict={x: val_x, keep_p: 1.0})
        test_x, test_y   = data.test_batch()
        test_x_weighted  = normalized.eval(feed_dict={x:test_x, keep_p:1.0})
        wv = WeightedVectors(method_name='SupervisedWeighting', from_dataset=data.name, from_category=FLAGS.cat,
                             trX=train_x_weighted, trY=train_y,
                             vaX=val_x_weighted, vaY=val_y,
                             teX=test_x_weighted, teY=test_y,
                             run_params_dic=run_params_dic)
        outname=FLAGS.outname
        if not outname:
            outname = '%s_C%d_F%d_H%d_lr%.5f_O%s_N%s_n%s_L%s_P%s_R%d.pickle' % \
                      (data.name[:3], FLAGS.cat, data.num_features(), FLAGS.hidden, FLAGS.lrate, FLAGS.optimizer,
                       FLAGS.normalize,FLAGS.forcepos, FLAGS.linidf, FLAGS.pretrain, FLAGS.run)
        wv.pickle(FLAGS.outdir, outname)
        print 'Weighted vectors saved at '+outname

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('dataset', '20newsgroups', 'Dataset in '+str(DatasetLoader.valid_datasets)+' (default 20newsgroups)')
    flags.DEFINE_float('fs', 0.1, 'Indicates the proportion of features to be selected (default 0.1).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 32, 'Size of the batches. Set to -1 to avoid batching (default 32).')
    flags.DEFINE_integer('hidden', 1000, 'Number of hidden nodes (default 1000).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)') #3e-4
    flags.DEFINE_string('optimizer', 'adam', 'Optimization algorithm in ["sgd", "adam", "rmsprop"] (default adam)')
    flags.DEFINE_boolean('normalize', True, 'Imposes normalization to the document vectors (default True)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_string('summariesdir', '../summaries', 'Directory for Tensorboard summaries (default "../summaries")')
    flags.DEFINE_string('pretrain', 'off', 'Pretrains the model parameters to mimic a given FS function, e.g., "infogain", "chisquare", "gss" (default "off")')
    flags.DEFINE_boolean('debug', False, 'Set to true for fast data load, and debugging')
    flags.DEFINE_boolean('linidf', True, 'Applies a non-linear function to the output of the idf-like part (default True)')
    flags.DEFINE_boolean('forcepos', True, 'Forces the idf-like part to be non-negative (default True)')
    flags.DEFINE_string('computation', 'tfidflike', 'Computation mode, see documentation (default tfidflike)')
    flags.DEFINE_string('plotmode', 'off', 'Select the mode of plotting for the the idf-like function learnt; available modes include:'
                                            '\n off: deactivated (default)'
                                            '\n show: shows the plot during training'
                                            '\n img: save images in pdf format'
                                            '\n vid: generates a video showing the evolution by steps')
    flags.DEFINE_string('plotdir', '../plot', 'Directory for plots, if --plot is True (default "../plot")')
    flags.DEFINE_string('outdir', '../vectors', 'Output dir for learned vectors (default "../vectors").')
    flags.DEFINE_string('outname', None, 'Output file name for learned vectors (default None --self defined with the rest of parameters).')
    flags.DEFINE_integer('run', 0, 'Specifies the number of run in case an experiment is to be replied more than once (default 0)')
    flags.DEFINE_string('notes', '', 'Informative notes to be stored within the pickle output file.')
    flags.DEFINE_string('resultcontainer', '../results.csv', 'If indicated, saves the result of the logistic regressor trained (default ../results.csv)')
    flags.DEFINE_integer('maxsteps', 100000, 'Maximun number of iterations (default 100000).')

    err_param_range('dataset', FLAGS.dataset, DatasetLoader.valid_datasets)
    err_param_range('cat', FLAGS.cat, valid_values=DatasetLoader.valid_catcodes[FLAGS.dataset])
    err_param_range('optimizer', FLAGS.optimizer, ['sgd', 'adam', 'rmsprop'])
    err_param_range('computation', FLAGS.computation, ['tfidflike','full'])
    err_param_range('pretrain',  FLAGS.pretrain,  ['off', 'infogain', 'chisquare', 'gss'])
    err_param_range('plotmode',  FLAGS.plotmode,  ['off', 'show', 'img', 'vid'])
    err_exit(FLAGS.fs <= 0.0 or FLAGS.fs > 1.0, 'Error: param fs should be in range (0,1]')
    err_exit(FLAGS.computation=='full' and FLAGS.plotmode!='off', 'Error: plot mode should be off when computation is set to full.')
    err_exit(FLAGS.computation == 'full' and FLAGS.pretrain != 'off', 'Error: pretrain mode should be off when computation is set to full.')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    if FLAGS.plotmode != 'show':
        os.environ['MATPLOTLIB_USE'] = 'Agg'
    from plot_function import PlotIdf
    tf.app.run()
