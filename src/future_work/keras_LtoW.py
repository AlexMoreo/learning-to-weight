import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from data.dataset_loader import TextCollectionLoader
from metrics import macroF1, microF1

def get_tpr_fpr_statistics(data):
    nC = data.num_categories()
    nF = data.num_features()
    matrix_4cell = data.get_4cell_matrix()
    feat_corr_info = np.array([[[matrix_4cell[c, f].tpr(), matrix_4cell[c, f].fpr()] for f in range(nF)] for c in range(nC)])
    info_by_feat = feat_corr_info.shape[-1]
    return feat_corr_info, info_by_feat

#dataload-----------------------------------------------------
data = TextCollectionLoader(dataset='reuters21578', rep_mode='dense', feat_sel=0.1, vectorizer='l1')
tpr_fpr_array, info_by_feat = get_tpr_fpr_statistics(data)

nF = data.num_features()
nC = data.num_categories()

#model--------------------------------------------------------
if data.classification == 'single-label':
    last_activation = 'softmax'
    loss = 'categorical_crossentropy'
elif data.classification == 'multi-label':
    last_activation = 'sigmoid'
    loss = 'binary_crossentropy'
else: raise ValueError('classification {} mode not allowed'.format(data.classification))

tf_filters=100

tf_input = Input(shape=(nF,), name='tf_input') #, sparse=True
#idf_input = Input(shape=(1,nF*info_by_feat), name='idf_input')

import keras.backend as K
import tensorflow as tf

# tf_in = Reshape(target_shape=(nF,1))(tf_input)
# h = TimeDistributed(Dense(units=tf_filters, activation='relu',use_bias=False))(tf_in)
# h = TimeDistributed(Dense(units=1, activation='relu',use_bias=False))(h)
# h = Reshape(target_shape=(nF,))(h)

h = Reshape(target_shape=(nF,1))(tf_input)
h = Dense(units=tf_filters, activation='relu',use_bias=False,
          kernel_constraint=None)(h) #see non_neg
h = Dense(units=1, activation='relu',use_bias=False,
          kernel_constraint=None)(h) #see non_neg
tf_weight = Reshape(target_shape=(nF,))(h)

# tf_in = Reshape(target_shape=(nF,1))(tf_input)
# h = Conv1D(filters=tf_filters,
#            kernel_size=1,
#            strides=1,
#            padding='valid',
#            activation='relu',
#            use_bias=False,
#            kernel_regularizer=None,
#            activity_regularizer=None,
#            kernel_constraint=None, #non_neg
#            bias_constraint=None
#            )(tf_in)
# h = Reshape(target_shape=(nF*tf_filters,1))(h)
# h = Conv1D(filters=1,
#            kernel_size=tf_filters,
#            strides=tf_filters,
#            padding='valid',
#            activation='relu',
#            use_bias=False,
#            kernel_regularizer=None,
#            activity_regularizer=None,
#            kernel_constraint=None, #non_neg
#            bias_constraint=None)(h)
# h = Flatten()(h)
# out = Dense(units=nC, activation=last_activation)(h)

#
# h = Dense(units=nF/2, input_dim=nF, activation='relu')(tf_input)
# h = Dropout(0.5)(h)
# h = Dense(units=nF/2, activation='relu')(h)
# h = Dropout(0.5)(h)


out = Dense(units=nC, activation=last_activation)(tf_weight)

model = Model(inputs=[tf_input], outputs=out)
tf_model = Model(inputs=[tf_input], outputs=tf_weight)

#sgd = SGD(lr=1.0, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
print(model)

#train--------------------------------------------------------
x_train, y_train = data.get_train_set()
x_val, y_val = data.get_validation_set()

earlystop=EarlyStopping(monitor='val_loss', patience=20)
#tensorboard=TensorBoard(log_dir='./logs', histogram_freq=1)
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=[earlystop])

#test--------------------------------------------------------

tf_domain = np.expand_dims(np.array([i*1.0/nF for i in range(nF)]), axis=0)
tf_domain_func = tf_model.predict(tf_domain)
tf_domain = tf_domain[0, ::100]
tf_domain_func = tf_domain_func[0, ::100]
plt.plot(tf_domain, tf_domain_func, 'r-')
plt.xlabel('|tf|_1')
plt.ylabel('DD(tf)')
plt.grid(True)
plt.show()

x_test, y_test = data.get_test_set()
y_predictions = np.rint(model.predict(x_test))

macro_f1=macroF1(y_test,y_predictions)
micro_f1=microF1(y_test,y_predictions)
print("Macro-F1={}, micro-F1={}".format(macro_f1,micro_f1))
