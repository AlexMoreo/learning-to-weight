from keras.models import Sequential
from keras.layers import Dense, Activation
from data.dataset_loader import TextCollectionLoader
from metrics import macroF1, microF1

#dataload-----------------------------------------------------
data = TextCollectionLoader(dataset='reuters21578', rep_mode='dense', feat_sel=0.1)
nF = data.num_features()
nC = data.num_categories()

#model--------------------------------------------------------
model = Sequential()
model.add(Dense(units=nF/2, input_dim=nF, activation='relu'))
model.add(Dense(units=nF/2, input_dim=nF, activation='relu'))
model.add(Dense(units=nF/2, input_dim=nF, activation='relu'))
model.add(Dense(units=nC, activation='softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#train--------------------------------------------------------
x_train, y_train = data.get_devel_set()
model.fit(x_train, y_train, epochs=20, batch_size=256)

#test--------------------------------------------------------
x_test, y_test = data.get_test_set()
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=256)
print(loss_and_metrics)
