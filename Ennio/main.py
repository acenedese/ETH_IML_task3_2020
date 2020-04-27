import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from threading import Lock, Thread
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Activation, Dropout, Flatten, LSTM, TimeDistributed


train = pd.DataFrame(pd.read_csv("../data/train.csv"))
X_test_raw = pd.DataFrame(pd.read_csv("../data/test.csv"))
sample = pd.DataFrame(pd.read_csv("../data/sample.csv"))

X_raw = pd.DataFrame(train['Sequence'])
Y = pd.DataFrame(train['Active'])

#------------------------------------
#--------- PARAMETERS ---------------
#------------------------------------
np.random.seed(seed=123)
epochs = 2
shuffle = False
split = 0.8
model_type = 'mlp' #choose between svm, mlp, cnn
#------------------------------------
#--------- PROCESS DATASET ----------
#------------------------------------
try:
    X = pd.DataFrame(pd.read_csv("../data/train_numeric.zip", compression='zip'))
    X_test = pd.DataFrame(pd.read_csv("../data/test_numeric.zip", compression='zip'))
except OSError:
    print('Numeric dataset not found, calculating')
    X_test_raw = pd.DataFrame(pd.read_csv("../data/test.csv"))
    amms = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q','C','U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    amms_dict = dict(zip(amms,range(21)))
    def encode_to_num(amm_str):
        code_num = np.zeros([len(amms)*4])
        for i in range(4):
            code_num[len(amms)*i + amms_dict[amm_str[i]]] = 1
        return pd.Series(dict(zip( sum([[a + str(i) for a in amms] for i in range(4)] ,[]) , code_num)))

    X = X_raw['Sequence'].apply(encode_to_num)
    X_test = X_test_raw['Sequence'].apply(encode_to_num)
    X.to_csv("../data/train_numeric.zip", compression='zip')
    X_test.to_csv("../data/test_numeric.zip", compression='zip')

#------------------------------------
#--------- BUILD THE MODEL ----------
#------------------------------------

if model_type == 'mlp':
    model = Sequential()
    model.add(Dense(400,
                        activation='relu',
                        input_dim=X.shape[1]))
    # model.add(Dropout(0.25))
    # model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(),
                     loss=keras.losses.binary_crossentropy,
                     metrics=[keras.metrics.categorical_accuracy])
elif model_type == "svm":
    model = svm.LinearSVC(C=1e-3, tol=1e-2, class_weight='balanced', verbose=0)
elif model_type == 'cnn':
    input_shape = (12, X.shape[1])

    model = Sequential()
    model.add(Conv1D(30, kernel_size=5,
                        activation='relu',
                        input_shape=input_shape))
    model.add(Conv1D(30, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(),
                     loss=keras.losses.binary_crossentropy,
                     metrics=[keras.metrics.categorical_accuracy])

#------------------------------------
#-------------- TRAIN ---------------
#------------------------------------

if shuffle:
    rd_permutation = np.random.permutation(X.index)
    X = X.reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))
    Y = pd.DataFrame(Y).reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))

#split
train_size = int(0.8*X.shape[0])
X_train = X.iloc[0:train_size,:]
Y_train = Y.iloc[0:train_size,:]
X_val = X.iloc[train_size+1:,:]
Y_val = Y.iloc[train_size+1:,:]

#train
if model_type == 'mlp' or model_type == 'cnn':
    model.fit(X_train, Y_train, epochs=epochs, verbose=1, class_weight='balanced')
    Y_val_pred = model.predict(X_val)
elif model_type == 'svm':
    model.fit(X_train, Y_train)
    Y_temp = np.array([model.decision_function(X_val)])
    Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()


#predict and eval
score = skmetrics.f1_score(Y_val, Y_val_pred>0.5)
print(score)