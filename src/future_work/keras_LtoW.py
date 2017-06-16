from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
if data.classification == 'single-label':
    last_activation = 'softmax'
    loss = 'categorical_crossentropy'
elif data.classification == 'multi-label':
    last_activation = 'sigmoid'
    loss = 'binary_crossentropy'
else: raise ValueError('classification {} mode not allowed'.format(data.classification))

#model--------------------------------------------------------
model = Sequential()
model.add(Dense(units=nF/2, input_dim=nF, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=nF/2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=nC, activation=last_activation))

#sgd = SGD(lr=1.0, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=loss,
              optimizer='rmsprop',
              metrics=['accuracy'])

#train--------------------------------------------------------
x_train, y_train = data.get_train_set()
x_val, y_val = data.get_validation_set()
earlystop=EarlyStopping(monitor='val_loss', patience=30)
tensorboard=TensorBoard(log_dir='./logs', histogram_freq=1)
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), callbacks=[earlystop,tensorboard])

#test--------------------------------------------------------
x_test, y_test = data.get_test_set()
y_predictions = np.rint(model.predict(x_test))

macro_f1=macroF1(y_test,y_predictions)
micro_f1=microF1(y_test,y_predictions)
print("Macro-F1={}, micro-F1={}".format(macro_f1,micro_f1))
