from time import gmtime, strftime
from classification_benchmark_multi import *
from feature_selection import tsr_function
from utils.tf_helpers import *

def get_tpr_fpr_statistics(data):
    nF = data.num_features()
    matrix_4cell = data.get_4cell_matrix()
    feat_corr_info = np.array([[matrix_4cell[0, f].tpr(), matrix_4cell[0, f].fpr()] for f in range(nF)])
    info_by_feat = feat_corr_info.shape[-1]
    return feat_corr_info, info_by_feat

def main(argv=None):
    err_exception(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    outname = FLAGS.outname
    if not outname:
        outname = 'LtoW_'+FLAGS.computation+('learnTF' if FLAGS.learntf==True else '' )+('%s_C%d_FS%.2f_H%d_lr%.5f_O%s_N%s_n%s_P%s_R%d.pickle' % \
                  (FLAGS.dataset[:3], FLAGS.cat, FLAGS.fs, FLAGS.hidden, FLAGS.lrate, FLAGS.optimizer,
                   FLAGS.normalize, FLAGS.forcepos, FLAGS.pretrain, FLAGS.run))

    # check if the vector has already been calculated
    err_exception(os.path.exists(join(FLAGS.outdir, outname)), 'Vector file %s already exists!' % outname)

    init_time = time.time()

    # old code
    data = TextCollectionLoader(dataset=FLAGS.dataset, vectorizer='count', rep_mode='dense', positive_cat=FLAGS.cat,feat_sel=FLAGS.fs)
    data.devel_vec = normalize(data.devel_vec, norm='l1', axis=1, copy=False)
    data.test_vec = normalize(data.test_vec, norm='l1', axis=1, copy=False)
    # updated code    #data = TextCollectionLoader(dataset=FLAGS.dataset, vectorizer='l1', rep_mode='dense', positive_cat=pos_cat_code, feat_sel=feat_sel, norm=None)


    print("|Tr|=%d [prev+ %f]" % (data.num_tr_documents(), data.train_class_prevalence(0)))
    print("|Val|=%d [prev+ %f]" % (data.num_val_documents(), data.valid_class_prevalence(0)))
    print("|Te|=%d [prev+ %f]" % (data.num_test_documents(), data.test_class_prevalence(0)))
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_categories())))

    print('Getting supervised correlations')
    feat_corr_info, info_by_feat = get_tpr_fpr_statistics(data)

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
            if FLAGS.learntf==True:
                print('learning the tf-like function')
                tf2h, _ = get_projection_weights([1, FLAGS.hiddentf], 'tf2h_weights', bias=False)
                h2tf, _ = get_projection_weights([FLAGS.hiddentf, 1], 'h2tf_weights', bias=False)
                tf_tensor = tf.reshape(x_raw, shape=[-1, 1])
                tf_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_tensor, tf2h)), keep_prob=keep_p)
                tf_like = tf.matmul(tf_hidden, h2tf)
                return tf.reshape(tf_like, shape=[-1, x_size])
            else:
                return tf.log(x_raw + 1)

        def local_idflike(info_arr):
            filter_weights, filter_biases = get_projection_weights([info_by_feat, 1, FLAGS.hidden], 'local_filter')
            proj_weights, proj_biases = get_projection_weights([FLAGS.hidden, 1], 'local_proj')
            n_results = info_arr.get_shape().as_list()[-1] / info_by_feat
            idf_tensor = tf.reshape(info_arr, shape=[1, -1, 1])
            conv = tf.nn.conv1d(idf_tensor, filters=filter_weights, stride=info_by_feat, padding='VALID')
            relu = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv, filter_biases)), keep_prob=keep_p)
            reshape = tf.reshape(relu, [n_results, FLAGS.hidden])
            proj = tf.nn.bias_add(tf.matmul(reshape, proj_weights), proj_biases)
            return tf.reshape(proj, [n_results])

        def global_idflike(info_arr):
            nf = data.num_features()
            info_arr_exp = tf.expand_dims(info_arr, 0)
            w1, b1 = get_projection_weights([nf * info_by_feat, nf/2], 'global_l1')
            w2, b2 = get_projection_weights([nf/2, nf], 'global_l2')
            h = tf.nn.relu(tf.matmul(info_arr_exp, w1) + b1)
            h = tf.nn.dropout(h, keep_prob=keep_p)
            return tf.nn.bias_add(tf.matmul(h, w2), b2)

        def idf_like(feat_info):
            if FLAGS.computation == 'local': idf_ = local_idflike(feat_info)
            elif FLAGS.computation == 'global': idf_ = global_idflike(feat_info)
            return tf.nn.sigmoid(idf_) if FLAGS.forcepos else idf_

        weighted_layer = tf.multiply(tf_like(x), idf_like(feat_info))
        normalized = tf.nn.l2_normalize(weighted_layer, dim=1) if FLAGS.normalize else weighted_layer
        logis_w, logis_b = get_projection_weights([data.num_features(), 1], 'logistic')
        logits = tf.squeeze(tf.nn.bias_add(tf.matmul(normalized, logis_w), logis_b))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        y_ = tf.nn.sigmoid(logits)
        prediction = tf.squeeze(tf.round(y_))  # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise

        end2end_optimizer  = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)
        logistic_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss, var_list=[logis_w, logis_b])

        #pre-learn the idf-like function as any feature selection function
        if FLAGS.pretrain != 'off':
            x_func = tf.placeholder(tf.float32, shape=[None, info_by_feat])
            y_func = tf.placeholder(tf.float32, shape=[None])
            tf.get_variable_scope().reuse_variables()
            if FLAGS.computation == 'local':
                idf_prediction = local_idflike(x_func)
                idf_loss = tf.reduce_mean(tf.square(tf.subtract(y_func, idf_prediction)))
                idf_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(idf_loss)

        saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts, dropout=False):
        x_, y_ = batch_parts
        return {x: x_, y: y_.reshape(-1), keep_p: drop_keep_p if dropout else 1.0}

    def predict(x, y, batch_size=FLAGS.batchsize):
        nD = x.shape[0]
        n_batches = nD // batch_size
        if nD % batch_size != 0: n_batches += 1
        predictions = []
        for i in range(n_batches):
            x_batch = x[i * batch_size : (i + 1) * batch_size]
            y_batch = y[i * batch_size : (i + 1) * batch_size]
            eval_dict = as_feed_dict((x_batch, y_batch), dropout=False)
            predictions.append(prediction.eval(feed_dict=eval_dict))
        predictions = np.concatenate(predictions)
        return predictions

    def weight_docs(x_docs, batch_size=FLAGS.batchsize):
        nD = x_docs.shape[0]
        n_batches = nD // batch_size
        if nD % batch_size != 0: n_batches += 1
        weights = []
        for i in range(n_batches):
            x_batch = x_docs[i * batch_size : (i + 1) * batch_size]
            weights.append(normalized.eval(feed_dict={x: x_batch, keep_p: 1.0}))
        weights = np.vstack(weights)
        return weights

    create_if_not_exists(FLAGS.checkpointdir)
    create_if_not_exists(FLAGS.summariesdir)
    create_if_not_exists(FLAGS.outdir)
    pc = data.devel_class_prevalence(0)

    def supervised_idf(tpr, fpr):
        if FLAGS.pretrain == 'off': return 0.0
        fsmethod = getattr(tsr_function, FLAGS.pretrain)
        return apply_tsr(tpr, fpr, pc, fsmethod)

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
        #tensorboard = TensorboardData()
        #tensorboard.open(FLAGS.summariesdir, 'sup_weight', session.graph)

        # pre-train -------------------------------------------------
        idf_steps = 0
        epsilon = 0.0003
        def idf_wrapper(x):
            return idf_prediction.eval(feed_dict={x_func: [x], keep_p: 1.0})
        plot = PlotIdf(FLAGS.plotmode, FLAGS.plotdir,
                       supervised_idf if FLAGS.pretrain!='off' else None, idf_wrapper, idf_points=feat_corr_info)
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
        logistic_optimization_phase = show_step*100
        best_f1 = 0.0
        log_steps = 0
        savedstep = -1
        for step in range(1,FLAGS.maxsteps):
            in_logistic_phase = FLAGS.pretrain!='off' and step < logistic_optimization_phase
            optimizer_ = logistic_optimizer if in_logistic_phase else end2end_optimizer
            tr_dict = as_feed_dict(data.train_batch(batch_size), dropout=True)
            _, l  = session.run([optimizer_, loss], feed_dict=tr_dict)
            l_ave += l
            log_steps += 1

            if step % show_step == 0:
                #sum = end2end_summaries.eval(feed_dict=tr_dict)
                #tensorboard.add_train_summary(sum, step+idf_steps)
                tr_phase = 'logistic' if in_logistic_phase else 'end2end'
                print('[step=%d][ep=%d][op=%s] loss=%.10f' % (step, data.epoch, tr_phase, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                val_x, val_y = data.get_validation_set()
                predictions = predict(val_x, val_y)
                acc, f1, p, r = evaluation_metrics(predictions, val_y)
                improves = f1 > best_f1
                if improves:
                    best_f1 = f1

                print('Validation acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f %s' % (acc, f1, p, r, ('[improves]' if improves else '[patience=%d]'%(early_stop_steps-last_improvement))))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savedstep=step+idf_steps
                    savemodel(session, savedstep, saver, FLAGS.checkpointdir, 'model')

                test_x, test_y = data.get_test_set()
                predictions = predict(test_x, test_y)
                acc, f1, p, r = evaluation_metrics(predictions, test_y)
                print('[Test acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f]' % (acc, f1, p, r))
                timeref = time.time()

            if FLAGS.plotmode=='vid' and step % plotsteps == 0:
                plot.plot(step=step+idf_steps)

            #early stop if does not improve after 10 validations
            if last_improvement >= early_stop_steps:
                if step < 10000 and best_f1==0.0:
                    print('Reinit weights! more patience...')
                    tf.initialize_all_variables().run()
                    last_improvement = 0
                else:
                    print('Early stop after %d validation steps without improvements' % last_improvement)
                    break
            if best_f1==1.0:
                print('Max validation score reached. End of training.')
                break

        #tensorboard.close()

        # output -------------------------------------------------
        print 'Test evaluation:'
        if savedstep>0:
            restore_checkpoint(saver, session, FLAGS.checkpointdir)
        if FLAGS.plotmode in ['img', 'show']: plot.plot(step=savedstep)
        test_x, test_y = data.get_test_set()
        predictions = predict(test_x, test_y)
        acc, f1, p, r = evaluation_metrics(predictions, test_y)
        print('Logistic Regression acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, p, r))

        # if indicated, saves the result of the current logistic regressor
        # if FLAGS.resultcontainer:
        #     results = BasicResultTable(FLAGS.resultcontainer)
        #     results.init_row_result('LogisticRegression-Internal', data, run=FLAGS.run)
        #     results.add_result_metric_scores(acc=acc, f1=f1, prec=p, rec=r,
        #                                      cont_table=contingency_table(predictions, eval_dict[y]), init_time=init_time)
        #     results.set_all(run_params_dic)
        #     results.commit()

        print 'Weighting documents'
        train_x, train_y = data.get_train_set()
        train_x_weighted = weight_docs(train_x)

        val_x, val_y   = data.get_validation_set()
        val_x_weighted = weight_docs(val_x)

        test_x, test_y   = data.get_test_set()
        test_x_weighted = weight_docs(test_x)

        vectorizer_name = 'LtoW_'+FLAGS.computation+('learnTF' if FLAGS.learntf==True else '')
        run_params_dic = {'num_features': data.num_features(),
                          'date': strftime("%d-%m-%Y", gmtime()),
                          'hiddensize': FLAGS.hidden,
                          'lrate': FLAGS.lrate,
                          'optimizer': FLAGS.optimizer,
                          'normalize': FLAGS.normalize,
                          'outmode': 'sigmoid' if FLAGS.forcepos else 'I',
                          'pretrain': FLAGS.pretrain,
                          'iterations': idf_steps + log_steps,
                          'notes': FLAGS.notes,
                          'run': FLAGS.run,
                          'learn_tf': FLAGS.learntf,
                          'learn_idf': True,
                          'learn_norm': False}
        wv = WeightedVectors(vectorizer=vectorizer_name, from_dataset=data.name, from_category=FLAGS.cat,
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

    flags.DEFINE_string('dataset', '20newsgroups', 'Dataset in ' + str(TextCollectionLoader.valid_datasets) + ' (default 20newsgroups)')
    flags.DEFINE_float('fs', 0.1, 'Indicates the proportion of features to be selected (default 0.1).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 100, 'Size of the batches. Set to -1 to avoid batching (default 100).')
    flags.DEFINE_integer('hidden', 1000, 'Number of hidden nodes (default 1000).')
    flags.DEFINE_integer('hiddentf', 100, 'Number of hidden nodes for tf-like param, ignored if learntf=False (default 100).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)') #3e-4
    flags.DEFINE_string('optimizer', 'adam', 'Optimization algorithm in ["sgd", "adam", "rmsprop"] (default adam)')
    flags.DEFINE_boolean('normalize', True, 'Imposes L2 normalization to the document vectors (default True)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_string('summariesdir', '../summaries', 'Directory for Tensorboard summaries (default "../summaries")')
    flags.DEFINE_string('pretrain', 'off', 'Pretrains the model parameters to mimic a given FS function, e.g., "infogain", "chisquare", "gss" (default "off")')
    flags.DEFINE_boolean('debug', False, 'Set to true for fast data load, and debugging')
    flags.DEFINE_boolean('forcepos', True, 'Forces the idf-like part to be non-negative (default True)')
    flags.DEFINE_string('computation', 'local', 'Computation mode, see documentation (default local)')
    flags.DEFINE_string('learntf', False, 'Learns the tf-like function (default False)')
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

    err_param_range('dataset', FLAGS.dataset, TextCollectionLoader.valid_datasets)
    err_param_range('cat', FLAGS.cat, valid_values=TextCollectionLoader.valid_catcodes[FLAGS.dataset])
    err_param_range('optimizer', FLAGS.optimizer, ['sgd', 'adam', 'rmsprop'])
    err_param_range('computation', FLAGS.computation, ['local','global'])
    err_param_range('pretrain',  FLAGS.pretrain,  ['off', 'infogain', 'chisquare', 'gss', 'rel_factor', 'idf'])
    err_param_range('plotmode',  FLAGS.plotmode,  ['off', 'show', 'img', 'vid'])
    err_exception(FLAGS.fs <= 0.0 or FLAGS.fs > 1.0, 'Error: param fs should be in range (0,1]')
    err_exception(FLAGS.computation == 'global' and FLAGS.plotmode != 'off', 'Error: plot mode should be off when computation is set to global.')
    err_exception(FLAGS.computation == 'global' and FLAGS.pretrain != 'off', 'Error: pretrain mode should be off when computation is set to global.')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    if FLAGS.plotmode != 'show':
        os.environ['MATPLOTLIB_USE'] = 'Agg'
    from utils.plot_function import PlotIdf
    tf.app.run()
