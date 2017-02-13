from time import gmtime, strftime

from src.classification_benchmark import *
from src.feature_selection import tsr_function


def main(argv=None):
    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    outname = FLAGS.outname
    if not outname:
        outname = '%s_C%d_F%d_IDF%s_R%d.pickle' % (
        FLAGS.dataset[:3], FLAGS.cat, FLAGS.fs, FLAGS.idflike, FLAGS.run)

    # check if the vector has already been calculated
    err_exit(os.path.exists(join(FLAGS.outdir, outname)), 'Vector file %s already exists!' % outname)

    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs
    data = DatasetLoader(dataset=FLAGS.dataset, vectorize='sublinear_tf', rep_mode='dense', positive_cat=pos_cat_code, feat_sel=feat_sel)
    print("|Tr|=%d [prev+ %f]" % (data.num_tr_documents(), data.train_class_prevalence()))
    print("|Val|=%d [prev+ %f]" % (data.num_val_documents(), data.valid_class_prevalence()))
    print("|Te|=%d [prev+ %f]" % (data.num_test_documents(), data.test_class_prevalence()))
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_categories())))

    print('Getting supervised correlations')
    idf_like_func = getattr(tsr_function, FLAGS.idflike)
    pc = data.devel_class_prevalence()
    sup = [data.feature_label_contingency_table(f, cat_label=1) for f in range(data.num_features())]
    feat_corr_info = [idf_like_func(sup_i.tpr(), sup_i.fpr(), pc) for sup_i in sup]

    x_size = data.num_features()
    batch_size = FLAGS.batchsize if FLAGS.batchsize!=-1 else data.num_tr_documents()

    # graph definition --------------------------------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None])
        tf_param = tf.Variable(tf.ones([1]), tf.float32)
        idf_param = tf.Variable(tf.ones([1]), tf.float32)
        feat_info = tf.constant(feat_corr_info, dtype=tf.float32)

        tf_like = tf.pow(x, tf.maximum(tf_param, 0.00001))
        idf_like = tf.pow(feat_info, tf.maximum(idf_param, 0.00001))
        tfidf_like = tf.mul(tf_like, idf_like)
        normalized = tf.nn.l2_normalize(tfidf_like, dim=1)
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
    create_if_not_exists(FLAGS.checkpointdir)
    create_if_not_exists(FLAGS.summariesdir)
    create_if_not_exists(FLAGS.outdir)

    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()

        # train -------------------------------------------------
        show_step = plotsteps = 10
        valid_step = show_step * 10
        last_improvement = 0
        early_stop_steps = 20
        l_ave=0.0
        timeref = time.time()
        best_f1, best_alpha, best_beta = 0.0, 1.0, 1.0
        savedstep = -1
        for step in range(1,FLAGS.maxsteps):
            x_,y_ = data.train_batch(batch_size)
            _, l, alpha, beta = session.run([optimizer, loss, tf_param, idf_param], feed_dict={x:x_, y:y_})
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][ep=%d][alpha=%.4f, beta=%.4f] loss=%.10f' % (step, data.epoch, alpha, beta, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                x_,y_ = data.val_batch()
                predictions = prediction.eval(feed_dict={x:x_, y:y_})
                acc, f1, p, r = evaluation_metrics(predictions, y_)
                improves = f1 > best_f1
                if improves:
                    best_f1, best_alpha, best_beta = f1, alpha, beta

                print('Validation acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f %s' % (acc, f1, p, r, ('[improves]' if improves else '')))
                last_improvement = 0 if improves else last_improvement + 1
                if improves:
                    savedstep=step
                    savemodel(session, savedstep, saver, FLAGS.checkpointdir, 'model')

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
        x_, y_ = data.test_batch()
        predictions = prediction.eval(feed_dict={x:x_, y:y_})
        acc, f1, p, r = evaluation_metrics(predictions, y_)
        print('Logistic Regression acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, p, r))

        run_params_dic = {'num_features': data.num_features(),
                          'date': strftime("%d-%m-%Y", gmtime()),
                          'lrate': FLAGS.lrate,
                          'optimizer': 'adam',
                          'normalize': True,
                          'pretrain': FLAGS.idflike,
                          'iterations': savedstep,
                          'notes': 'alpha=%.5f beta=%.5f'%(best_alpha, best_beta),
                          'run': FLAGS.run}

        # if indicated, saves the result of the current logistic regressor
        print 'Weighting documents'
        train_x, train_y = data.get_train_set()
        train_x_weighted = normalized.eval(feed_dict={x: train_x})
        val_x, val_y   = data.get_validation_set()
        val_x_weighted = normalized.eval(feed_dict={x: val_x})
        test_x, test_y   = data.test_batch()
        test_x_weighted  = normalized.eval(feed_dict={x:test_x})
        wv = WeightedVectors(vectorizer='a-b-'+FLAGS.idflike, from_dataset=data.name, from_category=FLAGS.cat,
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

    flags.DEFINE_string('dataset', '20newsgroups', 'Dataset in '+str(DatasetLoader.valid_datasets)+' (default 20newsgroups)')
    flags.DEFINE_float('fs', 0.1, 'Indicates the proportion of features to be selected (default 0.1).')
    flags.DEFINE_integer('cat', 0, 'Code of the positive category (default 0).')
    flags.DEFINE_integer('batchsize', 100, 'Size of the batches. Set to -1 to avoid batching (default 100).')
    flags.DEFINE_float('lrate', .005, 'Initial learning rate (default .005)') #3e-4
    flags.DEFINE_string('optimizer', 'adam', 'Optimization algorithm in ["sgd", "adam", "rmsprop"] (default adam)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_string('summariesdir', '../summaries', 'Directory for Tensorboard summaries (default "../summaries")')
    flags.DEFINE_string('outdir', '../vectors', 'Output dir for learned vectors (default "../vectors").')
    flags.DEFINE_string('idflike', None, 'IDF-like function.')
    flags.DEFINE_string('outname', None, 'Output file name for learned vectors (default None --self defined with the rest of parameters).')
    flags.DEFINE_integer('run', 0, 'Specifies the number of run in case an experiment is to be replied more than once (default 0)')
    flags.DEFINE_integer('maxsteps', 100000, 'Maximun number of iterations (default 100000).')

    err_param_range('dataset', FLAGS.dataset, DatasetLoader.valid_datasets)
    err_param_range('cat', FLAGS.cat, valid_values=DatasetLoader.valid_catcodes[FLAGS.dataset])
    err_exit(FLAGS.fs <= 0.0 or FLAGS.fs > 1.0, 'Error: param fs should be in range (0,1]')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    tf.app.run()
