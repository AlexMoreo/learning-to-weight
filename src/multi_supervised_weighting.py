import time
from time import gmtime, strftime

import numpy as np
from joblib import Parallel, delayed
from os.path import join

from data.dataset_loader import TextCollectionLoader
from data.weighted_vectors import WeightedVectors
from utils.helpers import *
from utils.tf_helpers import *
from utils.result_table import BasicResultTable
from utils.metrics import macroF1, microF1
from feature_selection.tsr_function import ContTable
from sklearn.preprocessing import normalize

def main(argv=None):
    err_exception(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    outname = FLAGS.outname
    confname = 'LtoW_' + ('Ltf' if FLAGS.learntf else '') + ('Lidf' if FLAGS.learnidf else '') + \
               ('Ln' if FLAGS.learnnorm else '') + (FLAGS.tfmode)
    if not outname:
        outname = '%s_%s_C%s_FS%.2f_H%d_lr%.5f_O%s_Ln%s_Ltf%s_Lidf%d_run%d.pickle' % \
                  (confname, FLAGS.dataset[:3], str(FLAGS.cat) if FLAGS.cat is not None else "multiclass", FLAGS.fs, FLAGS.hidden, FLAGS.lrate, 'adam',
                   FLAGS.learnnorm, FLAGS.learntf, FLAGS.learnidf, FLAGS.run)

    # check if the vector has already been calculated
    if not FLAGS.f:
        err_exception(os.path.exists(join(FLAGS.outdir, outname)), 'Vector file %s already exists!' % outname)

    init_time = time.time()
    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs

    l1_norm = False
    if FLAGS.tfmode == "tf":
        vectorizer = "tf"
        sublinear = False
    elif FLAGS.tfmode == "sublinear_tf":
        vectorizer = "tf"
        sublinear = True
    elif FLAGS.tfmode == "binary":
        vectorizer = "binary"
        sublinear = False
    elif FLAGS.tfmode == "l1":
        vectorizer = "tf"
        sublinear = False
        l1_norm = True
    else: raise ValueError("tfmode not supported")

    data = TextCollectionLoader(dataset=FLAGS.dataset, vectorizer=vectorizer, rep_mode='dense', positive_cat=pos_cat_code,
                                feat_sel=feat_sel, sublinear_tf=sublinear)

    if l1_norm:
        print('L1-normalize')
        data.devel_vec = normalize(data.devel_vec, norm='l1', axis=1, copy=False)
        data.test_vec  = normalize(data.test_vec, norm='l1', axis=1, copy=False)

    #max_term_frequency = 1.0
    max_term_frequency = np.amax(data.devel_vec)
    print("|Tr|=%d" % data.num_devel_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|C|=%d" % data.num_categories())
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_categories())))

    print('Getting supervised correlations')

    nC = data.num_categories()
    nF = data.num_features()

    matrix_4cell = data.get_4cell_matrix()
    feat_corr_info = np.array([[[matrix_4cell[c, f].tpr(), matrix_4cell[c, f].fpr()] for f in range(nF)] for c in range(nC)])
    info_by_feat = feat_corr_info.shape[-1]

    x_size = nF
    batch_size = FLAGS.batchsize if FLAGS.batchsize!=-1 else data.num_tr_documents()
    drop_keep_p = 0.9

    # graph definition --------------------------------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None, nC])
        freq_input = tf.placeholder(tf.float32, shape=[1,1])
        tprfpr_input = tf.placeholder(tf.float32, shape=[1,1,2])
        keep_p = tf.placeholder(tf.float32)
        var_p = tf.get_variable('p', shape=[1], initializer=tf.constant_initializer(2.0))
        #tf_pow = tf.get_variable('tf_pow', shape=[1], initializer=tf.constant_initializer(1.0))
        #tf_prod = tf.get_variable('tf_prod', shape=[1], initializer=tf.constant_initializer(1.0))
        #tf_offset = tf.get_variable('tf_sum', shape=[1], initializer=tf.constant_initializer(0.0))

        feat_info = tf.constant(feat_corr_info, dtype=tf.float32)

        def tf_like(x_raw, epsilon=1e-12):

            #mask = tf.ceil(x_raw)
            #return tf.mul(tf.add(tf.mul(tf.pow(x_raw, tf_pow), tf_prod), tf_offset), mask)
            #return tf.maximum(tf.add(tf.mul(tf.pow(x_raw, tf_pow), tf_prod), tf_offset), 0.0)
            #return tf.mul(tf.pow(x_raw, tf_pow), tf.maximum(tf_prod, epsilon))
            #tflike = tf.pow(tf.mul(x_raw, tf_prod), tf_pow)
            #tflike = tf.Print(tflike, [x_raw, tf_prod, tf_pow, tflike], message="x_raw, tf_prod, tf_pow, tflike")
            #return tflike
            #-------
            x_size = x_raw.get_shape().as_list()[1]
            width = 1
            in_channels = 1
            out_channels = 20
            filter_weights, filter_biases = get_projection_weights([width, in_channels, out_channels], 'local_tf_filter')
            proj_weights, proj_biases = get_projection_weights([out_channels, 1], 'local_tf_proj')
            tf_tensor = tf.reshape(x_raw, shape=[-1, x_size, 1])
            conv = tf.nn.conv1d(tf_tensor, filters=filter_weights, stride=1, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(conv+filter_biases), keep_prob=keep_p)
            reshape = tf.reshape(relu, [-1, out_channels])
            proj = tf.nn.relu(tf.matmul(reshape, proj_weights)+proj_biases)
            tflike = tf.reshape(proj, [-1, x_size])
            #tflike = tf.Print(tflike, [x_raw, tflike], message="tf-like: x_raw, tflike")
            #tf.get_variable_scope().reuse_variables()
            return tflike

        """This variant learns a function taking as inputs all pairs (tpr,fpr) from all categories, i.e.,
        assumes that the idf-like function is not unique for all categories"""
        def idf_like_2(info_arr):
            nC, nF, info_by_feat = info_arr.get_shape().as_list()
            in_channels = info_by_feat
            out_channels = FLAGS.hidden

            #filter shape=(filter_height, filter_width, in_channels, out_channels)
            filter_weights, filter_biases = get_projection_weights([nC, 1, in_channels, out_channels], 'local_idf_filter_2')
            proj_weights, proj_biases = get_projection_weights([out_channels, 1], 'local_idf_proj_2')

            #idf_tensor shape=(batch, height, width, channels) in NHWC format (default)
            idf_tensor = tf.reshape(info_arr, shape=[1, nC, nF, info_by_feat])
            strides = [1, 1, 1, 1]

            conv = tf.nn.conv2d(idf_tensor, filter=filter_weights, strides=strides, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [-1, out_channels])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            #proj = tf.matmul(reshape, proj_weights)
            proj_t = tf.reshape(proj, [1, nF])
            return proj_t

        """This variant assumes there is a idf-like function for all categories, and thus returns all idf-like values
        making no assumption on the later pooling"""
        def idf_like(info_arr):
            nC, nF, info_by_feat = info_arr.get_shape().as_list()
            in_channels = info_by_feat
            out_channels = FLAGS.hidden

            #filter shape=(filter_height, filter_width, in_channels, out_channels)
            filter_weights, filter_biases = get_projection_weights([1, 1, in_channels, out_channels], 'local_idf_filter')
            proj_weights, proj_biases = get_projection_weights([out_channels, 1], 'local_idf_proj')

            #idf_tensor shape=(batch, height, width, channels) in NHWC format (default)
            idf_tensor = tf.reshape(info_arr, shape=[1, nC, nF, info_by_feat])
            strides = [1, 1, 1, 1]

            conv = tf.nn.conv2d(idf_tensor, filter=filter_weights, strides=strides, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [-1, out_channels])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            #proj = tf.matmul(reshape, proj_weights)
            proj_t = tf.reshape(proj, [nC, nF])
            return proj_t

        def idf_pool(idf_like_for_cat):
            if nC == 1:
                return idf_like_for_cat
            idf_like_for_cat = tf.transpose(idf_like_for_cat)
            pool_hiddensize = nC / 2
            pool_w, pool_b = get_projection_weights([nC, pool_hiddensize], 'pool_proj_h')
            pool_hw, pool_hb = get_projection_weights([pool_hiddensize, 1], 'pool_proj_o')
            pooled_h = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(idf_like_for_cat, pool_w) , pool_b)), keep_prob=keep_p)
            pooled = tf.squeeze(tf.nn.bias_add(tf.matmul(pooled_h, pool_hw), pool_hb))
            #idf = tf.Print(idf, [idf, proj, relu, conv, proj_weights, filter_weights, filter_biases], message="idf-like:idf, proj, relu, conv, proj_weights, filter_weights, filter_biases ")
            return pooled

        def normalization_like(v, epsilon=1e-12):
            # L^p norm
            p = tf.maximum(var_p, epsilon)
            vv = tf.pow(tf.abs(v),p)
            sum_vv = tf.maximum(tf.reduce_sum(vv, 1, keep_dims=True), epsilon)
            den = tf.maximum(tf.pow(sum_vv, 1.0/p), epsilon)
            norm = tf.div(v,den)
            #norm = tf.Print(norm, [p, tf.reduce_sum(v, 1), sum_vv, den], message="p, sum(v), sum_vv, den: ")
            return norm

        def tf_like_nooffset(x):
            with tf.variable_scope("tflike_scope") as scope:
                tf_tensor = tf_like(x)
                scope.reuse_variables()
                tf_tensor = tf_tensor - tf_like(tf.zeros_like(x)) #removes the offset so that the tf factor for frequency 0 is 0
            return tf_tensor

        if FLAGS.learntf:
            tf_tensor = tf_like_nooffset(x)
        else:
            tf_tensor = x

        if FLAGS.learnidf:
            #idf_tensor = idf_pool(idf_like(feat_info))
            idf_tensor = idf_like_2(feat_info)
        else:
            idf_tensor = tf.ones_like(x)
        print "tf_tensor", tf_tensor.get_shape()
        print "idf_tensor", idf_tensor.get_shape()

        tf_ave = tf.reduce_mean(tf_tensor)
        idf_ave = tf.reduce_mean(idf_tensor)
        tfidf_tensor = tf.mul(tf_tensor, idf_tensor)

        if FLAGS.learnnorm:
            normalized = normalization_like(tfidf_tensor)
        else:
            normalized = tf.nn.l2_normalize(tfidf_tensor, dim=1)

        print "normalized", normalized.get_shape()
        logis_w, logis_b = get_projection_weights([nF, nC], 'logistic')
        #ffout = ff_multilayer(normalized, [2048, 1024], non_linear_function=tf.nn.relu, keep_prob=keep_p, name='ff_multilayer')
        #logis_w, logis_b = get_projection_weights([1024, 1], 'logistic')
        #logits = tf.nn.bias_add(tf.matmul(ffout, logis_w), logis_b)
        logits = tf.nn.bias_add(tf.matmul(normalized, logis_w), logis_b)
        print "logits", logits.get_shape()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y))
        #loss = tf.Print(loss, [loss], message="loss: ")
        print "loss", loss.get_shape()
        y_ = tf.nn.sigmoid(logits)
        prediction = tf.round(y_)  # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise

        #for plot
        #tf.get_variable_scope().reuse_variables()
        #tf_pred = tf_like_nooffset(freq_input)
        #idf_pred = idf_pool(idf_like(tprfpr_input))
        # idf_pred = idf_like_2(tprfpr_input)


        optimizer  = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)
        #gvs = optimizer.compute_gradients(loss, tf.trainable_variables())
        #capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
        #optimizer = optimizer.apply_gradients(capped_grads_and_vars)

        saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, dropout=False):
        x_, y_ = batch_parts
        return {x: x_, y: y_, keep_p: drop_keep_p if dropout else 1.0}

    create_if_not_exists(FLAGS.checkpointdir)
    create_if_not_exists(FLAGS.outdir)

    def tf_wrapper(freq):
        return tf_pred.eval(feed_dict={freq_input: [[freq]], keep_p: 1.0})[0][0]

    def idf_wrapper(tpr, fpr):
        return idf_pred.eval(feed_dict={tprfpr_input: [[[tpr,fpr]]], keep_p: 1.0})

    with tf.Session(graph=graph) as session:

        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()

        # train -------------------------------------------------
        show_step = 10
        valid_step = show_step * 10
        plotsteps = valid_step
        last_improvement = 0
        early_stop_steps = 20
        l_ave=0.0
        timeref = time.time()
        best_f1 = 0.0
        log_steps = 0
        savedstep = -1
        plot = PlotTfIdf(FLAGS.plotmode, FLAGS.plotdir, tf_wrapper, idf_wrapper, tf_points=[0,max_term_frequency], idf_points=None)#np.reshape(feat_corr_info, newshape=[-1,info_by_feat]))
        for step in range(1,FLAGS.maxsteps):
            tr_dict = as_feed_dict(data.train_batch(batch_size), dropout=True)
            #_, l, tf_po, tf_pr  = session.run([optimizer, loss, tf_pow, tf_prod], feed_dict=tr_dict)
            #_, l, p, tf_po, tf_pr, tf_of = session.run([optimizer, loss, var_p, tf_pow, tf_prod, tf_offset], feed_dict=tr_dict)
            _, l, p, tf_, idf_ = session.run([optimizer, loss, var_p, tf_ave, idf_ave], feed_dict=tr_dict)
            l_ave += l
            log_steps += 1

            if step % show_step == 0:
                #print('[step=%d][ep=%d][p %.3f tf:pow %.3f, tf:w %.3f tf:o: %.3f] loss=%.10f' % (step, data.epoch, p, tf_po, tf_pr, tf_of, l_ave / show_step))
                print('[step=%d][ep=%d][p %.3f tf %.3f idf %.3f] loss=%.10f' % (step, data.epoch, p, tf_, idf_, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                eval_dict = as_feed_dict(data.get_validation_set(), dropout=False)
                predictions = prediction.eval(feed_dict=eval_dict)
                macro_f1 = macroF1(predictions, eval_dict[y])
                micro_f1 = microF1(predictions, eval_dict[y])
                improves = macro_f1 > best_f1
                if improves:
                    best_f1 = macro_f1

                print('Validation macro_f1=%.3f, micro_f1=%.3f%s' % (macro_f1, micro_f1, ('[improves]' if improves else '')))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savedstep=step
                    savemodel(session, savedstep, saver, FLAGS.checkpointdir, 'model')

                #eval_dict = as_feed_dict(data.get_test_set(), dropout=False)
                #predictions = prediction.eval(feed_dict=eval_dict)
                #macro_f1 = macroF1(predictions, eval_dict[y])
                #micro_f1 = microF1(predictions, eval_dict[y])
                #print('[Test macro_f1=%.3f, micro_f1=%.3f]' % (macro_f1, micro_f1))
                timeref = time.time()

            if FLAGS.plotmode=='vid' and step % plotsteps == 0:
                plot.plot(step=step)
            if FLAGS.plotmode == 'show' and step % plotsteps == 0:
                plot.plot(step=step)

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
        if FLAGS.plotmode in ['img', 'show']: plot.plot(step=savedstep)
        #eval_dict = as_feed_dict(data.get_test_set(), dropout=False)
        #predictions = prediction.eval(feed_dict=eval_dict)
        #macro_f1 = macroF1(predictions, eval_dict[y])
        #micro_f1 = microF1(predictions, eval_dict[y])
        #print('Logistic Regression macro_f1=%.3f, micro_f1=%.3f' % (macro_f1, micro_f1))

        run_params_dic = {'num_features': data.num_features(),
                          'date': strftime("%d-%m-%Y", gmtime()),
                          'hiddensize': FLAGS.hidden,
                          'lrate': FLAGS.lrate,
                          'optimizer': 'adam',
                          'learntf': FLAGS.learntf,
                          'learnidf': FLAGS.learnidf,
                          'learnnorm': FLAGS.learnnorm,
                          'iterations': log_steps,
                          'notes': FLAGS.notes,
                          'run': FLAGS.run}

        # if indicated, saves the result of the current logistic regressor
        #if FLAGS.resultcontainer:
        #    results = BasicResultTable(FLAGS.resultcontainer)
        #    results.init_row_result('LogisticRegression-Internal', data, run=FLAGS.run)
        #    results.add_result_metric_scores(acc=-1, f1=macro_f1, prec=-1, rec=-1,
        #                                     cont_table=ContTable(-1,-1,-1,-1), init_time=init_time)
        #    results.set_all(run_params_dic)
        #    results.commit()

        print 'Weighting documents'
        def weight_vectors(raw_vectors, blocksize=1000):
            n_vec = raw_vectors.shape[0]
            weighted = []
            offset = 0
            while offset < n_vec:
                weighted.append(normalized.eval(feed_dict={x: raw_vectors[offset:offset+blocksize], keep_p: 1.0}))
                offset+=blocksize
            res = np.concatenate(weighted,axis=0)
            return res

        train_x, train_y = data.get_train_set()
        train_x_weighted = weight_vectors(train_x)
        val_x, val_y   = data.get_validation_set()
        val_x_weighted = weight_vectors(val_x)
        test_x, test_y   = data.get_test_set()
        test_x_weighted  = weight_vectors(test_x)
        wv = WeightedVectors(vectorizer=confname, from_dataset=data.name, from_category=FLAGS.cat,
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

    flags.DEFINE_string('dataset', '', 'Dataset in ' + str(TextCollectionLoader.valid_datasets) + ' (default none)')
    flags.DEFINE_float('fs', 0.1, 'Indicates the proportion of features to be selected (default 0.1).')
    flags.DEFINE_integer('cat', None, 'Code of the positive category (default None, i.e., multiclass setting).')
    flags.DEFINE_integer('batchsize', 100, 'Size of the batches. Set to -1 to avoid batching (default 100).')
    flags.DEFINE_integer('hidden', 1000, 'Number of hidden nodes (default 1000).')
    flags.DEFINE_float('lrate', .001, 'Initial learning rate (default .001)') #3e-4
    flags.DEFINE_string('tfmode', 'tf', 'Specifies the tf mode to select, in [tf, sublinear_tf, binary, l1] (default tf)')
    flags.DEFINE_boolean('learnnorm', True, 'Learns the normalization function (default True)')
    flags.DEFINE_boolean('learntf', True, 'Learns the tf-like function (default True)')
    flags.DEFINE_boolean('learnidf', True, 'Learns the idf-like function (default True)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_boolean('f', False, 'Forces the run, i.e., do not check if the vector has already been calculated.')
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
    #flags.DEFINE_string('resultcontainer', '../results.csv', 'If indicated, saves the result of the logistic regressor trained (default ../results.csv)')
    flags.DEFINE_integer('maxsteps', 100000, 'Maximun number of iterations (default 100000).')

    err_param_range('dataset', FLAGS.dataset, TextCollectionLoader.valid_datasets)
    err_param_range('tfmode', FLAGS.tfmode, ['tf','sublinear_tf','binary','l1'])
    err_param_range('cat', FLAGS.cat, valid_values=TextCollectionLoader.valid_catcodes[FLAGS.dataset] + [None])
    err_param_range('plotmode',  FLAGS.plotmode,  ['off', 'show', 'img', 'vid'])
    err_exception(FLAGS.fs <= 0.0 or FLAGS.fs > 1.0, 'Error: param fs should be in range (0,1]')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    if FLAGS.plotmode != 'show':
        os.environ['MATPLOTLIB_USE'] = 'Agg'
    from utils.plot_function import PlotTfIdf
    tf.app.run()
