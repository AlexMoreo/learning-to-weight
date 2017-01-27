from sklearn.datasets import fetch_20newsgroups
from reuters21578_parser import ReutersParser
from nltk.corpus import movie_reviews
import nltk
import sklearn
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from custom_vectorizers import *
from sklearn.externals.six.moves import urllib
from helpers import *
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from feature_selection_function import ContTable
from glob import glob
import cPickle as pickle
import tarfile
from os.path import join
from os import listdir
from feature_selection_function import *
import time
import scipy

class Dataset:
    def __init__(self, data, target, target_names):
        self.data = data
        self.target = target
        self.target_names = target_names

class DatasetLoader:
    valid_datasets = ['20newsgroups', 'reuters21578', 'ohsumed', 'movie_reviews', 'sentence_polarity', 'imdb']
    valid_vectorizers = ['tfcw', 'tfgr', 'tfidf', 'count', 'binary', 'hashing', 'sublinear_tfidf', 'sublinear_tf', 'tfchi2', 'tfig', 'tfrf', 'bm25']
    valid_repmodes = ['sparse', 'dense', 'sparse_index']
    valid_catcodes = {'20newsgroups':range(20), 'reuters21578':range(115), 'ohsumed':range(23), 'movie_reviews':[1], 'sentence_polarity':[1], 'imdb':[1]}
    def __init__(self, dataset, valid_proportion=0.2, vectorize='hashing', rep_mode='sparse', positive_cat=None, feat_sel=None):
        err_param_range('vectorize', vectorize, valid_values=DatasetLoader.valid_vectorizers)
        err_param_range('rep_mode', rep_mode, valid_values=DatasetLoader.valid_repmodes)
        err_param_range('dataset', dataset, valid_values=DatasetLoader.valid_datasets)
        err_exit(positive_cat is not None and positive_cat not in DatasetLoader.valid_catcodes[dataset], 'Error. Positive category not in scope.')
        self.name = dataset
        self.vectorize=vectorize
        self.rep_mode=rep_mode
        self.cat_vec_dic = dict()
        if dataset == '20newsgroups':
            self.devel = self.fetch_20newsgroups(subset='train')
            self.test  = self.fetch_20newsgroups(subset='test')
            self.classification = 'single-label'
        elif dataset == 'reuters21578':
            self.devel = self.fetch_reuters21579(subset='train')
            self.test  = self.fetch_reuters21579(subset='test')
            self.classification = 'multi-label'
        elif dataset == 'ohsumed':
            self.devel = self.fetch_ohsumed50k(subset='train')
            self.test = self.fetch_ohsumed50k(subset='test')
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
        self.devel_vec, self.test_vec = self._vectorize_documents()
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
        if feat_sel is not None:
            self.feature_selection(int(feat_sel*self.num_features()))

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
        self.devel.target = self.binarize_label_vector(self.devel.target, self.classification, self.positive_cat)
        self.test.target = self.binarize_label_vector(self.test.target, self.classification, self.positive_cat)
        self.devel.target_names = self.test.target_names = ['negative', 'positive']
        self.classification = 'binary' #informs that the category codes have been set to 0 for negative and 1 for positive
        self.cat_vec_dic = dict()

    def binarize_label_vector(self, labels, classification_type, pos_code=1):
        if classification_type in ['single-label']:
            return np.array([(1 if pos_code == label_code else 0) for label_code in labels])
        elif classification_type=='multi-label':
            return np.array([(1 if pos_code in doc_labels else 0) for doc_labels in labels])
        elif classification_type in ['binary', 'polarity']:
            return labels #nothing to do
        else:
            err_exit(err_msg='Error while binarizing category codes: unexpected classification type.')

    def feature_selection(self, feat_sel):
        if self.vectorize == 'hashing':
            print 'Warning: feature selection ommitted when hashing is activated'
            return
        if feat_sel is not None and feat_sel < self.devel_vec.shape[1]:
            print('Selecting %d most important features from %d features' % (feat_sel, self.num_features()))
            fs = SelectKBest(chi2, k=feat_sel)
            self.devel_vec = fs.fit_transform(self.devel_vec, self.devel.target)
            self.test_vec = fs.transform(self.test_vec)

    def __prevalence(self, inset, cat_label=1):
        return sum(1.0 for x in inset if x == cat_label) / len(inset)

    def devel_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.devel_indexes], cat_label)

    def train_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.train_indexes], cat_label)

    def valid_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.devel.target[self.valid_indexes], cat_label)

    def test_class_prevalence(self, cat_label=1):
        return self.__prevalence(self.test.target[self.test_indexes], cat_label)

    def _vectorize_documents(self):
        min_df = 1
        self.weight_getter = self._get_weights #default getter
        if self.vectorize == 'hashing':
            self.vectorizer = HashingVectorizer(n_features=2**16, stop_words='english', non_negative=True)
            self.weight_getter = self._get_none
        elif self.vectorize == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df)
        elif self.vectorize == 'sublinear_tfidf':
            self.vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=min_df)
        elif self.vectorize in ['tfig', 'tfgr', 'tfchi2', 'tfrf', 'tfcw']:
            binary_target = self.binarize_label_vector(self.devel.target, self.classification, self.positive_cat)
            if self.vectorize == 'tfig':
                self.vectorizer = TftsrVectorizer(binary_target, infogain, stop_words='english', sublinear_tf=True, min_df=min_df)
            elif self.vectorize == 'tfchi2':
                self.vectorizer = TftsrVectorizer(binary_target, chisquare, stop_words='english', sublinear_tf=True, min_df=min_df)
            elif self.vectorize == 'tfgr':
                self.vectorizer = TftsrVectorizer(binary_target, gainratio, stop_words='english', sublinear_tf=True, min_df=min_df)
            elif self.vectorize == 'tfrf':
                self.vectorizer = TftsrVectorizer(binary_target, rel_factor, stop_words='english', sublinear_tf=True, min_df=min_df)
            elif self.vectorize == 'tfcw':
                self.vectorizer = TftsrVectorizer(binary_target, conf_weight, stop_words='english', sublinear_tf=True, min_df=min_df)
        elif self.vectorize == 'sublinear_tf':
            self.vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, use_idf=False, min_df=min_df)
        elif self.vectorize == 'bm25':
            self.vectorizer = BM25(stop_words='english', min_df=min_df)
        elif self.vectorize == 'count':
            self.vectorizer = CountVectorizer(stop_words='english', min_df=min_df)
        elif self.vectorize == 'binary':
            self.vectorizer = CountVectorizer(stop_words='english', binary=True, min_df=min_df)
            self.weight_getter = self._get_none

        tini=time.time()
        devel_vec = self.vectorizer.fit_transform(self.devel.data)
        test_vec = self.vectorizer.transform(self.test.data)
        print("Vectorizer took %ds" % (time.time()-tini))
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
        if self.offset == 0 and self.epoch == 0: random.shuffle(self.train_indexes)
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

    def feature_label_contingency_table(self, feat_index, cat_label=1):
        feat_vec = self.devel_vec[:,feat_index] #TODO: cache also the feature-vectors
        if cat_label not in self.cat_vec_dic:
            binary_target = self.binarize_label_vector(self.devel.target, self.classification, cat_label)
            self.cat_vec_dic[cat_label] = set([i for i, label in enumerate(binary_target) if label == 1])
        feat_doc_set = set(feat_vec.nonzero()[0])
        return feature_label_contingency_table(self.cat_vec_dic[cat_label], feat_doc_set, self.num_devel_docs())

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
        return self.devel.target_names

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

    def fetch_ohsumed20k(self, data_path=None, subset='train'):
        _dataname = 'ohsumed_20k'
        if data_path is None:
            data_path = join(os.path.expanduser('~'), _dataname)
        create_if_not_exists(data_path)

        pickle_file = join(data_path, _dataname + '.' + subset + '.pickle')
        if not os.path.exists(pickle_file):
            DOWNLOAD_URL = ('http://disi.unitn.it/moschitti/corpora/ohsumed-first-20000-docs.tar.gz')
            archive_path = os.path.join(data_path, 'ohsumed-first-20000-docs.tar.gz')
            if not os.path.exists(archive_path):
                print("downloading file...")
                urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path)
            print("untarring ohsumed...")
            tarfile.open(archive_path, 'r:gz').extractall(data_path)
            untardir = 'ohsumed-first-20000-docs'

            target_names = []
            splitdata = dict({'training': set(), 'test': set()})
            classification = dict()
            content = dict()
            for split in ['training', 'test']:
                for cat_id in os.listdir(join(data_path, untardir, split)):
                    if cat_id not in target_names: target_names.append(cat_id)
                    for doc_id in os.listdir(join(data_path, untardir, split, cat_id)):
                        text_content = open(join(data_path, untardir, split, cat_id, doc_id), 'r').read()
                        if doc_id not in classification: classification[doc_id] = []
                        splitdata[split].add(doc_id)
                        classification[doc_id].append(cat_id)
                        if doc_id not in content: content[doc_id] = text_content
                target_names.sort()
                dataset = Dataset([], [], target_names)
                for doc_id in splitdata[split]:
                    dataset.data.append(content[doc_id])
                    dataset.target.append([target_names.index(cat_id) for cat_id in classification[doc_id]])
                splitname = 'train' if split == 'training' else 'test'
                pickle.dump(dataset, open(join(data_path, _dataname + '.' + splitname + '.pickle'), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

        return pickle.load(open(pickle_file, 'rb'))

    def fetch_ohsumed50k(self, data_path=None, subset='train', train_test_split=0.7):
        _dataname = 'ohsumed_50k'
        if data_path is None:
            data_path = join(os.path.expanduser('~'), _dataname)
        create_if_not_exists(data_path)

        pickle_file = join(data_path, _dataname + '.' + subset + str(train_test_split) + '.pickle')
        if not os.path.exists(pickle_file):
            DOWNLOAD_URL = ('http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz')
            archive_path = os.path.join(data_path, 'ohsumed-all-docs.tar.gz')
            if not os.path.exists(archive_path):
                print("downloading file...")
                urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path)
            untardir = 'ohsumed-all'
            if not os.path.exists(os.path.join(data_path, untardir)):
                print("untarring ohsumed...")
                tarfile.open(archive_path, 'r:gz').extractall(data_path)


            target_names = []
            doc_classes = dict()
            class_docs = dict()
            content = dict()
            doc_ids = set()
            for cat_id in os.listdir(join(data_path, untardir)):
                target_names.append(cat_id)
                class_docs[cat_id] = []
                for doc_id in os.listdir(join(data_path, untardir, cat_id)):
                    doc_ids.add(doc_id)
                    text_content = open(join(data_path, untardir, cat_id, doc_id), 'r').read()
                    if doc_id not in doc_classes: doc_classes[doc_id] = []
                    doc_classes[doc_id].append(cat_id)
                    if doc_id not in content: content[doc_id] = text_content
                    class_docs[cat_id].append(doc_id)
            target_names.sort()
            print('Read %d different documents' % len(doc_ids))

            splitdata = dict({'train':[], 'test':[]})
            for cat_id in target_names:
                free_docs = [d for d in class_docs[cat_id] if (d not in splitdata['train'] and d not in splitdata['test'])]
                if len(free_docs) > 0:
                    split_point = int(math.floor(len(free_docs)*train_test_split))
                    splitdata['train'].extend(free_docs[:split_point])
                    splitdata['test'].extend(free_docs[split_point:])
            for split in ['train', 'test']:
                dataset = Dataset([], [], target_names)
                for doc_id in splitdata[split]:
                    dataset.data.append(content[doc_id])
                    dataset.target.append([target_names.index(cat_id) for cat_id in doc_classes[doc_id]])
                pickle.dump(dataset, open(join(data_path, _dataname + '.' + split + str(train_test_split) + '.pickle'), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

        return pickle.load(open(pickle_file, 'rb'))


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

            #debug esto

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



