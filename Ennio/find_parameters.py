import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from threading import Lock, Thread
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Activation, Dropout, Flatten, LSTM, TimeDistributed
from keras.regularizers import l2

train = pd.DataFrame(pd.read_csv("../data/train.csv"))
X_test_raw = pd.DataFrame(pd.read_csv("../data/test.csv"))
sample = pd.DataFrame(pd.read_csv("../data/sample.csv"))

X_raw = pd.DataFrame(train['Sequence'])
Y = pd.DataFrame(train['Active'])

# ------------------------------------
# --------- PARAMETERS ---------------
# ------------------------------------
np.random.seed(seed=400)
shuffle = False
split = 0.8

# ------------------------------------
# --------- PROCESS DATASET ----------
# ------------------------------------
try:
    X = pd.DataFrame(pd.read_csv("../data/train_numeric.zip", compression='zip', index_col=0))
    X_test = pd.DataFrame(pd.read_csv("../data/test_numeric.zip", compression='zip', index_col=0))
except OSError:
    print('Numeric dataset not found, calculating')
    X_test_raw = pd.DataFrame(pd.read_csv("../data/test.csv"))
    amms = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    amms_dict = dict(zip(amms, range(21)))


    def encode_to_num(amm_str):
        code_num = np.zeros([len(amms) * 4])
        for i in range(4):
            code_num[len(amms) * i + amms_dict[amm_str[i]]] = 1
        return code_num


    X = np.array(X_raw['Sequence'].apply(encode_to_num).values.tolist())
    X_test = np.array(X_test_raw['Sequence'].apply(encode_to_num).values.tolist())
    columns = sum([[a + str(i) for a in amms] for i in range(4)], [])
    X = pd.DataFrame(data=X, columns=columns)
    X_test = pd.DataFrame(data=X_test, columns=columns)
    X.to_csv("../data/train_numeric.zip", compression='zip')
    X_test.to_csv("../data/test_numeric.zip", compression='zip')

# ------------------------------------
# -------------- TRAIN ---------------
# ------------------------------------

if shuffle:
    rd_permutation = np.random.permutation(X.index)
    X = X.reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))
    Y = pd.DataFrame(Y).reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))

# split
train_size = int(0.8 * X.shape[0])
X_train = X.iloc[0:train_size, :]
Y_train = Y.iloc[0:train_size, :]
X_val = X.iloc[train_size + 1:, :]
Y_val = Y.iloc[train_size + 1:, :]

# define class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(Y),
    np.array(Y).flatten())

layer_sizes = [500, 550, 600, 650, 700]
alphas = [1e-10, 5e-10, 1e-9, 5e-9, 8e-9]
N = 3
matrix = pd.DataFrame(data=0, columns=alphas, index=layer_sizes)
lock = Lock()


def calc_one_profile(alpha, layer_size):
    score = 0
    model = MLPClassifier(hidden_layer_sizes=(layer_size,), activation='relu',  # 50, reg=8e-3 ==> 0.893
                          solver='adam', verbose=1, tol=3e-9, alpha=alpha, max_iter=1000)
    # train
    model.fit(X_train, Y_train)

    # predict and eval
    lock.acquire()
    score = score + skmetrics.f1_score(Y_val, model.predict(X_val) > 0.5) / N
    matrix.loc[layer_size, alpha] = matrix.loc[layer_size, alpha] + score
    matrix.to_csv("../data/parameters_9.csv")
    lock.release()

threads = []
for alpha in alphas:
    for layer_size in layer_sizes:
        for i in range(N):
            threads = threads + [
                Thread(target=calc_one_profile,
                       args=(alpha, layer_size),
                       name='Thread_' + str(layer_size) + '_' + str(alpha) + '_' + str(i))]
            threads[-1].start()

for thread in threads:
    thread.join()
