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
from joblib import Parallel, delayed
import multiprocessing

#TODO: if pow works, check with relu, and check with 1+log(tf)
#TODO: check the mul in tf_like, change to pow; debug (el mul no hace absolutamente nada, porque al normalizar se pierde)
#TODO: add information to sup_info, e.g., idf, ig, ptp ptn pfp pfn
#TODO: ConfWeight
#TODO: check out the separability index
#TODO: add a new non-linear classifier
#TODO: parallelize supervised info vector
#TODO: convolution on the supervised feat-cat statistics + freq (L1) + prob C (could be useful for non binary class)
def main(argv=None):
    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    outname = FLAGS.outname
    if not outname:
        outname = '%s_C%d_FS%.2f_H%d_lr%.5f_O%s_N%s_n%s_P%s_R%d.pickle' % \
                  (FLAGS.dataset[:3], FLAGS.cat, FLAGS.fs, FLAGS.hidden, FLAGS.lrate, FLAGS.optimizer,
                   FLAGS.normalize, FLAGS.forcepos, FLAGS.pretrain, FLAGS.run)

    # check if the vector has already been calculated
    err_exit(os.path.exists(join(FLAGS.outdir, outname)), 'Vector file %s already exists!' % outname)

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
    num_cores = multiprocessing.cpu_count()
    #sup = Parallel(n_jobs=num_cores, backend="threading")(
    #    delayed(data.feature_label_contingency_table)(f, cat_label=1) for f in range(data.num_features()))
    sup = [data.feature_label_contingency_table(f, cat_label=1) for f in range(data.num_features())]
    feat_corr_info = [[sup_i.tpr(), sup_i.fpr()] for sup_i in sup]

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
        var_p = tf.Variable(tf.ones([1]), tf.float32)

        feat_info = tf.constant(np.concatenate(feat_corr_info), dtype=tf.float32)

        def tf_like(x_raw):
            #tf_pow = tf.get_variable('tf_pow', shape=[1], initializer=tf.constant_initializer(1.0))
            #tf_prod = tf.get_variable('tf_prod', shape=[1], initializer=tf.constant_initializer(1.0))
            #tf_offset = tf.get_variable('tf_sum', shape=[1], initializer=tf.constant_initializer(0.0))

            #return tf.add(tf.mul(tf.pow(x_raw, tf_pow), tf_prod), tf_offset)
            #-------
            width = 1
            in_channels = 1
            out_channels = FLAGS.hidden / 20
            filter_weights, filter_biases = get_projection_weights([width, in_channels, out_channels], 'local_tf_filter')
            proj_weights, proj_biases = get_projection_weights([out_channels, 1], 'local_tf_proj')
            tf_tensor = tf.reshape(x_raw, shape=[-1, x_size, 1])
            conv = tf.nn.conv1d(tf_tensor, filters=filter_weights, stride=1, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [-1, out_channels])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            return tf.reshape(proj, [-1, x_size])

        def idf_like(info_arr):
            width = info_by_feat
            in_channels = 1
            out_channels = FLAGS.hidden
            filter_weights, filter_biases = get_projection_weights([width, in_channels, out_channels], 'local_idf_filter')
            proj_weights, proj_biases = get_projection_weights([out_channels, 1], 'local_idf_proj')
            n_results = info_arr.get_shape().as_list()[-1] / info_by_feat
            idf_tensor = tf.reshape(info_arr, shape=[1, -1, 1])
            conv = tf.nn.conv1d(idf_tensor, filters=filter_weights, stride=info_by_feat, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [n_results, out_channels])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            return tf.reshape(proj, [n_results])

        def normalization_like(v):
            # L^p norm
            print 'shape'
            print v.get_shape()
            vv = tf.pow(tf.abs(v),var_p)
            print vv.get_shape()
            sum_vv = tf.reduce_sum(vv, 1, keep_dims=True)
            print sum_vv.get_shape()
            den = tf.pow(sum_vv, 1.0/var_p)
            print den.get_shape()
            return tf.div(v,den)

        normalized = normalization_like(tf.mul(tf_like(x), idf_like(feat_info)))
        logis_w, logis_b = get_projection_weights([data.num_features(), 1], 'logistic')
        logits = tf.squeeze(tf.nn.bias_add(tf.matmul(normalized, logis_w), logis_b))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y))

        y_ = tf.nn.sigmoid(logits)
        prediction = tf.squeeze(tf.round(y_))  # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise

        optimizer  = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)

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

    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()

        # train -------------------------------------------------
        show_step = plotsteps = 1
        valid_step = show_step * 10
        last_improvement = 0
        early_stop_steps = 20
        l_ave=0.0
        timeref = time.time()
        best_f1 = 0.0
        log_steps = 0
        savedstep = -1
        for step in range(1,FLAGS.maxsteps):
            tr_dict = as_feed_dict(data.train_batch(batch_size), dropout=True)
            _, l  = session.run([optimizer, loss], feed_dict=tr_dict)
            l_ave += l
            log_steps += 1

            if step % show_step == 0:
                print('[step=%d][ep=%d] loss=%.10f' % (step, data.epoch, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                eval_dict = as_feed_dict(data.val_batch(), dropout=False)
                predictions = prediction.eval(feed_dict=eval_dict)
                acc, f1, p, r = evaluation_metrics(predictions, eval_dict[y])
                improves = f1 > best_f1
                if improves:
                    best_f1 = f1

                print('Validation acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f %s' % (acc, f1, p, r, ('[improves]' if improves else '')))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savedstep=step
                    savemodel(session, savedstep, saver, FLAGS.checkpointdir, 'model')

                eval_dict = as_feed_dict(data.test_batch(), dropout=False)
                predictions = prediction.eval(feed_dict=eval_dict)
                acc, f1, p, r = evaluation_metrics(predictions, eval_dict[y])
                print('[Test acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f]' % (acc, f1, p, r))
                timeref = time.time()

            #if FLAGS.plotmode=='vid' and step % plotsteps == 0:
            #    plot.plot(step=step)

            #early stop if not improves after 10 validations
            if last_improvement >= early_stop_steps:
                print('Early stop after %d validation steps without improvements' % last_improvement)
                break
            if best_f1==1.0:
                print('Max validation score reached. End of training.')
                break

        # output -------------------------------------------------
        print 'Test evaluation:'
        if savedstep>0:
            restore_checkpoint(saver, session, FLAGS.checkpointdir)
        #if FLAGS.plotmode in ['img', 'show']: plot.plot(step=savedstep)
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
                          'iterations': log_steps,
                          'notes': FLAGS.notes,
                          'run': FLAGS.run}

        # if indicated, saves the result of the current logistic regressor
        if FLAGS.resultcontainer:
            results = ReusltTable(FLAGS.resultcontainer)
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
        wv = WeightedVectors(vectorizer='full_LtoW', from_dataset=data.name, from_category=FLAGS.cat,
                             trX=train_x_weighted, trY=train_y,
                             vaX=val_x_weighted, vaY=val_y,
                             teX=test_x_weighted, teY=test_y,
                             run_params_dic=run_params_dic)
        wv.pickle(FLAGS.outdir, outname)
        print 'Weighted vectors saved at '+outname

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('dataset', '', 'Dataset in '+str(DatasetLoader.valid_datasets)+' (default none)')
    flags.DEFINE_float('fs', 0.1, 'Indicates the proportion of features to be selected (default 0.1).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 100, 'Size of the batches. Set to -1 to avoid batching (default 100).')
    flags.DEFINE_integer('hidden', 1000, 'Number of hidden nodes (default 1000).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)') #3e-4
    flags.DEFINE_string('optimizer', 'adam', 'Optimization algorithm in ["sgd", "adam", "rmsprop"] (default adam)')
    flags.DEFINE_boolean('normalize', True, 'Imposes normalization to the document vectors (default True)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_string('summariesdir', '../summaries', 'Directory for Tensorboard summaries (default "../summaries")')
    flags.DEFINE_string('pretrain', 'off', 'Pretrains the model parameters to mimic a given FS function, e.g., "infogain", "chisquare", "gss" (default "off")')
    flags.DEFINE_boolean('debug', False, 'Set to true for fast data load, and debugging')
    flags.DEFINE_boolean('forcepos', True, 'Forces the idf-like part to be non-negative (default True)')
    flags.DEFINE_string('computation', 'local', 'Computation mode, see documentation (default local)')
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
    err_param_range('computation', FLAGS.computation, ['local','global'])
    err_param_range('pretrain',  FLAGS.pretrain,  ['off', 'infogain', 'chisquare', 'gss', 'rel_factor', 'idf'])
    err_param_range('plotmode',  FLAGS.plotmode,  ['off', 'show', 'img', 'vid'])
    err_exit(FLAGS.fs <= 0.0 or FLAGS.fs > 1.0, 'Error: param fs should be in range (0,1]')
    err_exit(FLAGS.computation == 'global' and FLAGS.plotmode!='off', 'Error: plot mode should be off when computation is set to global.')
    err_exit(FLAGS.computation == 'global' and FLAGS.pretrain != 'off', 'Error: pretrain mode should be off when computation is set to global.')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    if FLAGS.plotmode != 'show':
        os.environ['MATPLOTLIB_USE'] = 'Agg'
    from plot_function import PlotIdf
    tf.app.run()