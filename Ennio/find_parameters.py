import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import threading
from threading import Thread

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Activation, Dropout, Flatten, LSTM, TimeDistributed
from keras.regularizers import l2

if __name__ == '__main__':
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
    def read_and_build_dataset():
        try:
            X = pd.DataFrame(pd.read_csv("../data/train_numeric.zip", compression='zip', index_col=0))
        except OSError:
            print('Numeric dataset not found, calculating')
            amms = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
            amms_dict = dict(zip(amms, range(21)))


            def encode_to_num(amm_str):
                code_num = np.zeros([len(amms) * 4])
                for i in range(4):
                    code_num[len(amms) * i + amms_dict[amm_str[i]]] = 1
                return code_num


            X = np.array(X_raw['Sequence'].apply(encode_to_num).values.tolist())
            columns = sum([[a + str(i) for a in amms] for i in range(4)], [])
            X = pd.DataFrame(data=X, columns=columns)
            X.to_csv("../data/train_numeric.zip", compression='zip')

        # if shuffle:
        #     rd_permutation = np.random.permutation(X.index)
        #     X = X.reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))
        #     Y = pd.DataFrame(Y).reindex(rd_permutation).set_index(np.arange(0, Y.shape[0], 1))

        # split
        train_size = int(0.8 * X.shape[0])
        X_train = X.iloc[0:train_size, :]
        Y_train = Y.iloc[0:train_size, :]
        X_val = X.iloc[train_size + 1:, :]
        Y_val = Y.iloc[train_size + 1:, :]
        return X_train, Y_train, X_val, Y_val


    # ------------------------------------
    # -------------- TRAIN ---------------
    # ------------------------------------

    # define class weights
    # class_weights = class_weight.compute_class_weight(
    #     'balanced',
    #     np.unique(Y),
    #     np.array(Y).flatten())

    layer_sizes = [600, 700]
    alphas = [1e-10, 5e-10]
    N = 3
    matrix = pd.DataFrame(data=0, columns=alphas, index=layer_sizes)

    import multiprocessing
    from multiprocessing import Process

    lock = threading.Lock()


    def calc_one_profile(alpha, layer_size):
        X_train, Y_train, X_val, Y_val = read_and_build_dataset()
        score = 0
        model = MLPClassifier(hidden_layer_sizes=(layer_size,), activation='relu',  # 50, reg=8e-3 ==> 0.893
                              solver='adam', verbose=0, tol=3e-9, alpha=alpha, max_iter=1000)

        t = True
        if t:
            name = threading.current_thread().name
        else:
            name = multiprocessing.current_process().name

        # train
        print(name + "started training\n")
        model.fit(X_train, np.ravel(Y_train))
        print(name + "terminated training\n")

        # predict and eval
        lock.acquire()
        score = score + skmetrics.f1_score(Y_val, model.predict(X_val) > 0.5) / N
        matrix.loc[layer_size, alpha] = matrix.loc[layer_size, alpha] + score
        matrix.to_csv("../data/parameters_9.csv")
        lock.release()

    work_flows = []
    for alpha in alphas:
        for layer_size in layer_sizes:
            for i in range(N):
                work_flows = work_flows + [
                    Thread(target=calc_one_profile,
                           args=(alpha, layer_size),
                           name='Work_flow_' + str(layer_size) + '_' + str(alpha) + '_' + str(i))]
                work_flows[-1].start()

    for work_flow in work_flows:
        work_flow.join()
