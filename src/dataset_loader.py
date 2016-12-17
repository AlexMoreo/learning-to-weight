from sklearn.datasets import fetch_20newsgroups
from reuters21578_parser import ReutersParser
from nltk.corpus import movie_reviews
import nltk
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.six.moves import urllib
from helpers import *
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from feature_selection_function import ContTable
from glob import glob
import cPickle as pickle
import tarfile
from os.path import join
from os import listdir

class Dataset:
    def __init__(self, data, target, target_names):
        self.data = data
        self.target = target
        self.target_names = target_names

class DatasetLoader:
    valid_datasets = ['20newsgroups', 'reuters21578', 'movie_reviews', 'sentence_polarity', 'imdb']
    valid_vectorizers = ['hashing', 'tfidf', 'count', 'binary', 'sublinear_tfidf', 'sublinear_tf']
    valid_repmodes = ['sparse', 'dense', 'sparse_index']
    valid_catcodes = {'20newsgroups':range(20), 'reuters21578':range(115), 'movie_reviews':[1], 'sentence_polarity':[1], 'imdb':[1]}
    def __init__(self, dataset, valid_proportion=0.1, vectorize='hashing', rep_mode='sparse', positive_cat=None, feat_sel=None):
        err_param_range('vectorize', vectorize, valid_values=DatasetLoader.valid_vectorizers)
        err_param_range('rep_mode', rep_mode, valid_values=DatasetLoader.valid_repmodes)
        err_param_range('dataset', dataset, valid_values=DatasetLoader.valid_datasets)
        err_exit(positive_cat is not None and positive_cat not in DatasetLoader.valid_catcodes[dataset],
                 'Error. Positive category not in scope.')
        self.name = dataset
        self.vectorize=vectorize
        self.rep_mode=rep_mode
        if dataset == '20newsgroups':
            self.devel = self.fetch_20newsgroups(subset='train')
            self.test  = self.fetch_20newsgroups(subset='test')
            self.classification = 'single-label'
        elif dataset == 'reuters21578':
            self.devel = self.fetch_reuters21579(subset='train')
            self.test  = self.fetch_reuters21579(subset='test')
            self.classification = 'multi-label'
        elif dataset == 'movie_reviews':
            self.devel = self.fetch_movie_reviews(subset='train')
            self.test = self.fetch_movie_reviews(subset='test')
            self.classification = 'polarity'
        elif dataset == 'sentence_polarity':
            self.devel = self.fetch_sentence_polarity(subset='train')
            self.test = self.fetch_sentence_polarity(subset='test')
            self.classification = 'polarity'
        elif dataset == 'imdb':
            self.devel = self.fetch_IMDB(subset='train')
            self.test = self.fetch_IMDB(subset='test')
            self.classification = 'polarity'
        self.epoch = 0
        self.offset = 0
        self.positive_cat = positive_cat
        self.devel_vec, self.test_vec = self._vectorize_documents(vectorize)
        self.devel_indexes = self._get_doc_indexes(self.devel_vec)
        self.test_indexes  = self._get_doc_indexes(self.test_vec)
        if self.rep_mode=='dense':
            self.devel_vec = self.devel_vec.todense()
            self.test_vec = self.test_vec.todense()
            self._batch_getter = self._dense_batch_getter
        elif self.rep_mode=='sparse':
            # already sparse representation
            self._batch_getter = self._sparse_batch_getter
        else: #sparse index
            # already sparse representation
            self._batch_getter = self._sparse_index_batch_getter
        if self.positive_cat is not None:
            print('Binarize towards positive category %s' % self.devel.target_names[self.positive_cat])
            self.binarize_classes()
            self.divide_train_val_evenly(valid_proportion=valid_proportion)
            self.feature_selection(feat_sel)
        self.cat_vec_dic = dict()

    # Ensures the train and validation splits to approximately preserve the original devel prevalence.
    # In extremely imbalanced cases, the train set is guaranteed to have some positive examples
    def divide_train_val_evenly(self, valid_proportion=0.5, pos_cat_code=1):
        pos_indexes = [ind for ind in self.devel_indexes if self.devel.target[ind] == pos_cat_code]
        neg_indexes = [ind for ind in self.devel_indexes if self.devel.target[ind] != pos_cat_code]
        pos_split_point = int(math.ceil(len(pos_indexes)*(1.0-valid_proportion)))
        neg_split_point = int(math.ceil(len(neg_indexes)*(1.0-valid_proportion)))
        self.train_indexes = np.array(pos_indexes[:pos_split_point] + neg_indexes[:neg_split_point])
        self.valid_indexes = np.array(pos_indexes[pos_split_point:] + neg_indexes[neg_split_point:])

    # change class codes: positive class = 1, all others = 0, and set category names to 'positive' or 'negative'
    def binarize_classes(self):
        if self.classification == 'polarity': return
        self.devel.target_names = self.test.target_names = ['negative', 'positive']
        def __binarize_codes_single_label(target, pos_code):
            target[target == pos_code] = -1
            target[target != -1] = 0
            target[target == -1] = 1
        def __binarize_codes_multi_label(target, pos_code):
            return np.array([(1 if pos_code in labels else 0) for labels in target])
        if self.classification == 'single-label':
            __binarize_codes_single_label(self.devel.target, self.positive_cat)
            __binarize_codes_single_label(self.test.target,  self.positive_cat)
        elif self.classification == 'multi-label':
            self.devel.target = __binarize_codes_multi_label(self.devel.target, self.positive_cat)
            self.test.target = __binarize_codes_multi_label(self.test.target, self.positive_cat)

    def feature_selection(self, feat_sel):
        if self.vectorize == 'hashing':
            print 'Warning: feature selection ommitted when hashing is activated'
            return
        if feat_sel is not None:
            print('Selecting %d most important features' % feat_sel)
            fs = SelectKBest(chi2, k=feat_sel)
            self.devel_vec = fs.fit_transform(self.devel_vec, self.devel.target)
            self.test_vec = fs.transform(self.test_vec)
            print('Done')

    def __prevalence(self, inset, cat_label=1):
        return sum(1.0 for x in inset if x == cat_label) / len(inset)

    def devel_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.devel_indexes], cat_label)

    def train_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.train_indexes], cat_label)

    def valid_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.valid_indexes], cat_label)

    def test_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.test_indexes], cat_label)

    def _vectorize_documents(self, vectorize):
        if vectorize == 'hashing':
            vectorizer = HashingVectorizer(n_features=2**16, stop_words='english', non_negative=True)
            self.weight_getter = self._get_none
        elif vectorize == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english')
            self.weight_getter = self._get_weights
        elif vectorize == 'sublinear_tfidf':
            vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
            self.weight_getter = self._get_weights
        elif vectorize == 'sublinear_tf':
            vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, use_idf=False)
            self.weight_getter = self._get_weights
        elif vectorize == 'count':
            vectorizer = CountVectorizer(stop_words='english')
            self.weight_getter = self._get_weights
        else: #binary
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            self.weight_getter = self._get_none
        devel_vec = vectorizer.fit_transform(self.devel.data)
        test_vec = vectorizer.transform(self.test.data)
        #sorting the indexes simplifies the creation of sparse tensors a lot
        devel_vec.sort_indices()
        test_vec.sort_indices()
        return devel_vec, test_vec

    #virtually remove invalid documents (documents without any non-zero feature)
    def _get_doc_indexes(self, vector_set):
        all_indexes = np.arange(vector_set.shape[0])
        return [i for i in all_indexes if vector_set[i].nnz>0]

    def get_index_values(self, batch):
        num_indices = len(batch.nonzero()[0])
        indices = batch.nonzero()[0].reshape(num_indices, 1)
        values = batch.nonzero()[1]
        return indices, values

    def _get_none(self, batch):
        return []

    def _get_weights(self, batch):
        return batch.data

    def _sparse_batch_getter(self, batch):
        return batch

    def _sparse_index_batch_getter(self, batch):
        indices, values = self.get_index_values(batch)
        weights = self.weight_getter(batch)
        return indices, values, weights

    def _dense_batch_getter(self, batch):
        return batch

    def _null_batch_getter(self, batch):
        indices, values = self.get_index_values(batch)
        weights = self.weight_getter(batch)
        return indices, values, weights

    def __get_set(self, vec_set, target, indexes):
        repr = self._batch_getter(vec_set[indexes])
        labels = target[indexes]
        return repr, labels

    def get_devel_set(self):
        return self.__get_set(self.devel_vec, self.devel.target, self.devel_indexes)

    def get_train_set(self):
        return self.__get_set(self.devel_vec, self.devel.target, self.train_indexes)

    def get_validation_set(self):
        return self.__get_set(self.devel_vec, self.devel.target, self.valid_indexes)

    def get_test_set(self):
        return self.__get_set(self.test_vec, self.test.target, self.test_indexes)

    def train_batch(self, batch_size=64):
        to_pos = min(self.offset + batch_size, self.num_tr_documents())
        batch = self.devel_vec[self.train_indexes[self.offset:to_pos]]
        batch_rep = self._batch_getter(batch)
        labels = self.devel.target[self.train_indexes[self.offset:to_pos]]
        self.offset += batch_size
        if self.offset >= self.num_tr_documents():
            self.offset = 0
            self.epoch += 1
            random.shuffle(self.train_indexes)
        return batch_rep, labels

    def val_batch(self):
        return self.get_validation_set()

    def test_batch(self):
        return self.get_test_set()

    def feat_sup_statistics(self, feat_index, cat_label=1):
        feat_vec = self.devel_vec[:,feat_index]
        if cat_label not in self.cat_vec_dic:
            self.cat_vec_dic[cat_label] = [(1 if x==cat_label else 0) for x in self.devel.target]
        cat_vec  = self.cat_vec_dic[cat_label]
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(feat_vec)):
            if feat_vec[i]>0:
                if cat_vec[i] > 0: tp += 1
                else: fp += 1
            else:
                if cat_vec[i] > 0: fn += 1
                else: tn += 1
        return ContTable(tp=tp, tn=tn, fp=fp, fn=fn)

    def num_devel_docs(self):
        return len(self.devel_indexes)

    def num_categories(self):
        return len(np.unique(self.devel.target))

    def num_features(self):
        return self.devel_vec.shape[1]

    def num_tr_documents(self):
        return len(self.train_indexes)

    def num_val_documents(self):
        return len(self.valid_indexes)

    def num_test_documents(self):
        return len(self.test_indexes)

    def get_categories(self):
        return np.unique(self.devel.target_names)

    def fetch_20newsgroups(self, data_path=None, subset='train'):
        if data_path is None:
            data_path = os.path.join(get_data_home(), '20newsgroups')
            create_if_not_exists(data_path)
        _20news_pickle_path = os.path.join(data_path, "20newsgroups." + subset + ".pickle")
        if not os.path.exists(_20news_pickle_path):
            metadata = ('headers', 'footers', 'quotes')
            dataset = fetch_20newsgroups(subset=subset, remove=metadata)
            pickle.dump(dataset, open(_20news_pickle_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            dataset = pickle.load(open(_20news_pickle_path, 'rb'))
        return dataset

    def fetch_reuters21579(self, data_path=None, subset='train'):
        if data_path is None:
            data_path = os.path.join(get_data_home(), 'reuters')
        reuters_pickle_path = os.path.join(data_path, "reuters." + subset + ".pickle")
        if not os.path.exists(reuters_pickle_path):
            parser = ReutersParser()
            for filename in glob(os.path.join(data_path, "*.sgm")):
                parser.parse(open(filename, 'rb'))
            # index category names with a unique numerical code (only considering categories with training examples)
            tr_categories = np.unique(np.concatenate([doc['topics'] for doc in parser.tr_docs])).tolist()

            def pickle_documents(docs, subset):
                for doc in docs:
                    doc['topics'] = [tr_categories.index(t) for t in doc['topics'] if t in tr_categories]
                pickle_docs = {'categories': tr_categories, 'documents': docs}
                pickle.dump(pickle_docs, open(os.path.join(data_path, "reuters." + subset + ".pickle"), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
                return pickle_docs

            pickle_tr = pickle_documents(parser.tr_docs, "train")
            pickle_te = pickle_documents(parser.te_docs, "test")
            print('Empty docs %d' % parser.empty_docs)
            requested_subset = pickle_tr if subset == 'train' else pickle_te
        else:
            requested_subset = pickle.load(open(reuters_pickle_path, 'rb'))

        data = [(u'{title}\n{body}\n{unproc}'.format(**doc), doc['topics']) for doc in requested_subset['documents']]
        text_data, topics = zip(*data)
        return Dataset(data=text_data, target=topics, target_names=requested_subset['categories'])

    def __distribute_evenly(self, pos_docs, neg_docs, target_names):
        data = pos_docs + neg_docs
        target = [1] * len(pos_docs) + [0] * len(neg_docs)
        # to prevent all positive (negative) documents to be placed together, we distribute them evenly
        data, target = shuffle_tied(data, target, random_seed=0)
        return Dataset(data=np.array(data),
                       target=np.array(target),
                       target_names=target_names)

    def __process_posneg_dataset(self, pos_docs, neg_docs, train_test_split):
        random.shuffle(pos_docs)
        random.shuffle(neg_docs)
        tr_pos, te_pos = split_at(pos_docs, train_test_split)
        tr_neg, te_neg = split_at(neg_docs, train_test_split)
        target_names = ['negative', 'positive']
        train = self.__distribute_evenly(tr_pos, tr_neg, target_names)
        test = self.__distribute_evenly(te_pos, te_neg, target_names)
        return train, test

    def __pickle_dataset(self, train, test, path, name, posfix=''):
        pickle.dump(train, open(join(path, name+'.train' + posfix + '.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test, open(join(path, name+'.test' + posfix + '.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def fetch_movie_reviews(self, subset='train', data_path=None, train_test_split=0.7):
        if data_path is None:
            data_path = join(nltk.data.path[0], 'corpora')

        _posfix=str(train_test_split)
        _dataname='movie_reviews'
        moviereviews_pickle_path = os.path.join(data_path, _dataname + '.' + subset + _posfix + '.pickle')
        if not os.path.exists(moviereviews_pickle_path):
            documents = dict([(cat, []) for cat in ['neg', 'pos']])
            [documents[i.split('/')[0]].append(' '.join([w for w in movie_reviews.words(i)])) for i in movie_reviews.fileids()]

            train, test = self.__process_posneg_dataset(pos_docs=documents['pos'], neg_docs=documents['neg'], train_test_split=train_test_split)
            self.__pickle_dataset(train, test, data_path,_dataname,_posfix)

            return train if subset == 'train' else test
        else:
            return pickle.load(open(moviereviews_pickle_path, 'rb'))

    def fetch_sentence_polarity(self, subset='train', data_path=None, train_test_split=0.7):
        if data_path is None:
            data_path = join(os.path.expanduser('~'), 'sentiment_data')
        create_if_not_exists(data_path)

        _posfix=str(train_test_split)
        _dataname='sentence_polarity'
        sentpolarity_pickle_path = join(data_path, _dataname + '.' + subset + _posfix + '.pickle')

        if not os.path.exists(sentpolarity_pickle_path):
            DOWNLOAD_URL = ('https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz')
            archive_path = os.path.join(data_path, 'rt-polaritydata.tar.gz')
            print("downloading file...")
            urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path)
            print("untarring sentiment dataset...")
            tarfile.open(archive_path, 'r:gz').extractall(data_path)
            positive_sentences = [unicode(s, 'ISO-8859-1') for s in open(os.path.join(data_path, 'rt-polaritydata', 'rt-polarity.pos'), 'r')]
            negative_sentences = [unicode(s, 'ISO-8859-1') for s in open(os.path.join(data_path, 'rt-polaritydata', 'rt-polarity.neg'), 'r')]
            train, test = self.__process_posneg_dataset(positive_sentences, negative_sentences, train_test_split)
            self.__pickle_dataset(train, test, data_path, _dataname, _posfix)

            return train if subset == 'train' else test
        else:
            return pickle.load(open(sentpolarity_pickle_path, 'rb'))

    def fetch_IMDB(self, subset='train', data_path=None):
        if data_path is None:
            data_path = join(os.path.expanduser('~'), 'sentiment_data')
        create_if_not_exists(data_path)

        _dataname = 'imdb'
        imdb_pickle_file = join(data_path, _dataname+'.'+subset+'.pickle')
        if not os.path.exists(imdb_pickle_file):
            DOWNLOAD_URL = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
            archive_path = os.path.join(data_path, 'aclImdb_v1.tar.gz')
            if not os.path.exists(archive_path):
                print("downloading file...")
                urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path)
            print("untarring sentiment dataset...")
            tarfile.open(archive_path, 'r:gz').extractall(data_path)

            data = dict()
            for split in ['train', 'test']:
                if split not in data: data[split] = dict()
                for polarity in ['pos', 'neg']:
                    if polarity not in data[split]: data[split][polarity] = []
                    reviews_path = join(data_path,'aclImdb',split,polarity)
                    for review in listdir(reviews_path):
                        review_txt = open(join(reviews_path, review),'r').read()
                        data[split][polarity].append(review_txt)

            target_names = ['negative', 'positive']

            train = self.__distribute_evenly(data['train']['pos'], data['train']['neg'], target_names)
            test = self.__distribute_evenly(data['test']['pos'], data['test']['neg'], target_names)
            self.__pickle_dataset(train, test, data_path, _dataname)

            return train if subset=='train' else test

        else:
            return pickle.load(open(imdb_pickle_file, 'rb'))


