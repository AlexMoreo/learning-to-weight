import time
from time import gmtime, strftime

import numpy as np
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
    if not outname:
        outname = 'simple_%s_C%s_FS%.2f_lr%.5f_O%s_N%s_R%d.pickle' % \
                  (FLAGS.dataset[:3], str(FLAGS.cat) if FLAGS.cat is not None else "multiclass", FLAGS.fs, FLAGS.lrate, FLAGS.optimizer,
                   FLAGS.normalize, FLAGS.run)

    # check if the vector has already been calculated
    if not FLAGS.f:
        err_exception(os.path.exists(join(FLAGS.outdir, outname)), 'Vector file %s already exists!' % outname)

    init_time = time.time()
    pos_cat_code = FLAGS.cat
    feat_sel = FLAGS.fs
    data = TextCollectionLoader(dataset=FLAGS.dataset, vectorizer='tf', rep_mode='dense', positive_cat=pos_cat_code,
                                feat_sel=feat_sel,
                                sublinear_tf=False)
    print('L1-normalize')
    data.devel_vec = normalize(data.devel_vec, norm='l1', axis=1, copy=False)
    data.test_vec  = normalize(data.test_vec, norm='l1', axis=1, copy=False)
    #max_term_frequency = 1.0
    #max_term_frequency = np.amax(data.devel_vec)
    print("|Tr|=%d" % data.num_devel_documents())
    print("|Te|=%d" % data.num_test_documents())
    print("|C|=%d" % data.num_categories())
    print("|V|=%d" % data.num_features())
    print("|C|=%d, %s" % (data.num_categories(), str(data.get_categories())))

    print('Getting supervised correlations')

    nC = data.num_categories()
    nF = data.num_features()

    x_size = nF
    batch_size = FLAGS.batchsize if FLAGS.batchsize!=-1 else data.num_tr_documents()

    # graph definition --------------------------------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None, nC])
        idfparams = tf.get_variable('idfparams', shape=[x_size], initializer=tf.constant_initializer(1.0))


        normalized = tf.nn.l2_normalize(tf.mul(x, idfparams), dim=1)
        logis_w, logis_b = get_projection_weights([nF, nC], 'logistic')
        logits = tf.matmul(normalized, logis_w) + logis_b
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y))
        y_ = tf.nn.sigmoid(logits)
        prediction = tf.round(y_)  # label the prediction as 0 if the P(y=1|x) < 0.5; 1 otherwhise

        optimizer  = tf.train.AdamOptimizer(learning_rate=FLAGS.lrate).minimize(loss)

        saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------
    def as_feed_dict(batch_parts):
        x_, y_ = batch_parts
        return {x: x_, y: y_}

    create_if_not_exists(FLAGS.checkpointdir)
    create_if_not_exists(FLAGS.outdir)

    with tf.Session(graph=graph) as session:
        n_params = count_trainable_parameters()
        print ('Number of model parameters: %d' % (n_params))
        tf.initialize_all_variables().run()

        # train -------------------------------------------------
        show_step = 100
        valid_step = show_step * 10
        last_improvement = 0
        early_stop_steps = 20
        l_ave=0.0
        timeref = time.time()
        best_f1 = 0.0
        savedstep = -1
        for step in range(1,FLAGS.maxsteps):
            tr_dict = as_feed_dict(data.train_batch(batch_size))
            _, l = session.run([optimizer, loss], feed_dict=tr_dict)
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][ep=%d] loss=%.10f' % (step, data.epoch, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                print ('Average time/step %.4fs' % ((time.time()-timeref)/valid_step))
                eval_dict = as_feed_dict(data.get_validation_set())
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

                timeref = time.time()

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
        eval_dict = as_feed_dict(data.get_test_set())
        predictions = prediction.eval(feed_dict=eval_dict)
        macro_f1 = macroF1(predictions, eval_dict[y])
        micro_f1 = microF1(predictions, eval_dict[y])
        print('Logistic Regression macro_f1=%.3f, micro_f1=%.3f' % (macro_f1, micro_f1))

        run_params_dic = {'num_features': data.num_features(),
                          'date': strftime("%d-%m-%Y", gmtime()),
                          'hiddensize': -1,
                          'lrate': FLAGS.lrate,
                          'optimizer': FLAGS.optimizer,
                          'normalize': FLAGS.normalize,
                          'nonnegative': -1,
                          'pretrain': -1,
                          'iterations': savedstep,
                          'notes': FLAGS.notes,
                          'run': FLAGS.run}

        print 'Weighting documents'
        def weight_vectors(raw_vectors, blocksize=1000):
            n_vec = raw_vectors.shape[0]
            weighted = []
            offset = 0
            while offset < n_vec:
                weighted.append(normalized.eval(feed_dict={x: raw_vectors[offset:offset+blocksize]}))
                offset+=blocksize
            res = np.concatenate(weighted,axis=0)
            return res

        train_x, train_y = data.get_train_set()
        train_x_weighted = normalized.eval(feed_dict={x: train_x})
        val_x, val_y   = data.get_validation_set()
        val_x_weighted = normalized.eval(feed_dict={x: val_x})
        test_x, test_y   = data.get_test_set()
        test_x_weighted  =  normalized.eval(feed_dict={x: test_x})
        wv = WeightedVectors(vectorizer='simple_LtoW', from_dataset=data.name, from_category=FLAGS.cat,
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
    flags.DEFINE_float('lrate', .001, 'Initial learning rate (default .001)') #3e-4
    flags.DEFINE_string('optimizer', 'adam', 'Optimization algorithm in ["sgd", "adam", "rmsprop"] (default adam)')
    flags.DEFINE_boolean('normalize', True, 'Imposes normalization to the document vectors (default True)')
    flags.DEFINE_string('checkpointdir', '../model', 'Directory where to save the checkpoints of the model parameters (default "../model")')
    flags.DEFINE_boolean('f', False, 'Forces the run, i.e., do not check if the vector has already been calculated.')
    flags.DEFINE_string('outdir', '../vectors', 'Output dir for learned vectors (default "../vectors").')
    flags.DEFINE_string('outname', None, 'Output file name for learned vectors (default None --self defined with the rest of parameters).')
    flags.DEFINE_integer('run', 0, 'Specifies the number of run in case an experiment is to be replied more than once (default 0)')
    flags.DEFINE_string('notes', '', 'Informative notes to be stored within the pickle output file.')
    flags.DEFINE_integer('maxsteps', 100000, 'Maximun number of iterations (default 100000).')

    err_param_range('dataset', FLAGS.dataset, TextCollectionLoader.valid_datasets)
    err_param_range('cat', FLAGS.cat, valid_values=TextCollectionLoader.valid_catcodes[FLAGS.dataset] + [None])
    err_param_range('optimizer', FLAGS.optimizer, ['sgd', 'adam', 'rmsprop'])

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    tf.app.run()
