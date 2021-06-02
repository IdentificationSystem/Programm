from mpl_toolkits.mplot3d import Axes3D  # For Basic ploting
from sklearn.preprocessing import StandardScaler  # Preprocessing
from sklearn import preprocessing  # Preprocessing
from random import seed
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB  # import gaussian naive bayes model
from sklearn.tree import DecisionTreeClassifier  # import Decision tree classifier
from sklearn import metrics  # Import scikit - learn metrics module for accuracy calculation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt# plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I / O(e.g.pd.read_csv)

for dirname, _, filenames in os.walk("C:\\Users\\sanch\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\tensorflow\\tensorflow-for-poets-2-master\\tf_files\\test_data"):
    for filename in filenames:
        print(os.path.join(dirname))


def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    # create first hidden layer
    model.add(tf.keras.Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(tf.keras.Dense(lyrs[i], activation=act))
        # add dropout, default is none
    model.add(tf.keras.Dropout(dr))
    # create output layer
    model.add(tf.keras.Dense(1, activation='sigmoid'))  # output layer
    model.complete(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def create_model(lyrs=[5], act='linear', opt='Adam', dr=0.0):
    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    # create first hidden layer
    model.add(tf.keras.Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(tf.keras.Dense(lyrs[i], activation=act))
        # add dropout, default is none
    model.add(tf.keras.Dropout(dr))
    # create output layer
    model.add(tf.keras.Dense(1, activation='sigmoid'))  # output layer
    model.complete(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

    model = create_model()
    print(model.summary())

    # train model on full train set, witch 80/20 CV split
    trainig = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = np.mean(trainig.history['val_acc'])
    print("\n%s: %.2f%%" % ('val_acc', val_acc * 100))

    model = create_model()
    print(model.summary())

    # train model on full train set, witch 80/20 CV split
    trainig = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = np.mean(trainig.history['val_acc'])
    print("\n%s: %.2f%%" % ('val_acc', val_acc * 100))

def create_model(lyrs=[10], act='linear', opt='Adam', dr=0.0):
    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    # create first hidden layer
    model.add(tf.keras.Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(tf.keras.Dense(lyrs[i], activation=act))
        # add dropout, default is none
    model.add(tf.keras.Dropout(dr))
    # create output layer
    model.add(tf.keras.Dense(1, activation='sigmoid'))  # output layer
    model.complete(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

    model = create_model()
    print(model.summary())

    # train model on full train set, witch 80/20 CV split
    trainig = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = np.mean(trainig.history['val_acc'])
    print("\n%s: %.2f%%" % ('val_acc', val_acc * 100))

    # create model
    model = tf.keras.KerasClassifier(build_nf=create_model, verbose=0)

    # define the grid search parameters
    batch_size = [16, 32, 64]
    epochs = [50, 100]
    param_grid = dict(batch_size=batch_size, ephochs=epochs)

    # search the grid
    grid = tf.keras.GridSearchCV(estimator=model,
                                 param_grid=param_grid,
                                 cv=3,
                                 verbose=2)  # include n_jobs=-1 if you are using CPU
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    nRowsRead = None  # specify No.of row. 'None' for whole data
    # test_mosaic.scv may have more rows in reality, but we are only loading / previewing the first 1000 rows
    train_data = pd.read_csv("C:\\Users\\sanch\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\tensorflow\\tensorflow-for-poets-2-master\\tf_files\\test_data", delimiter=',', nrows=nRowsRead)
    nRow, nCol = train_data.shape
    print(f'Yhere are {nRow} rows and {nCol} columns')

    X_train = train_data.drop('Label', axis=1)
    X_test = test_data.drop('Label', axis=1)
    y_train = train_data['Label']
    y_test = test_data['Label']
