import pandas as pd
import numpy as np
import scipy.signal as sig
from glob import glob
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential, load_model
import matplotlib.pyplot as plt

column_names = ['hour', 'minute', 'second', 'micsec', 'horAcc', 'vertAcc']


def constructSignal(bearing_folder, axis='horAcc'):
    signal = []
    fileformat = 'acc_{:05d}.csv'
    nrFiles = len(glob(bearing_folder + 'acc*csv'))
    print(bearing_folder)
    for nr in range(1, nrFiles + 1):
        filename = bearing_folder + fileformat.format(nr)
        values = pd.read_csv(filename, names=column_names)[axis]
        signal = np.append(signal, np.array(values))
    return signal


def constructTimeSignal(bearing_folder, axis='horAcc'):
    signal = []
    fileformat = 'temp_{:05d}.csv'
    nrFiles = len(glob(bearing_folder + 'temp*csv'))
    print(bearing_folder)
    for nr in range(1, nrFiles + 1):
        filename = bearing_folder + fileformat.format(nr)
        values = pd.read_csv(filename, names=column_names)[axis]
        signal = np.append(signal, np.array(values))
    return signal


def getSpectro(signal):
    freq_array, segment_times, spectrogram = sig.spectrogram(x=signal, fs=25600, nperseg=2560, noverlap=0)
    return spectrogram.T


# maak train en test data
def splitTestAndTrain(np_spec, test_frac=0.1):
    # test_frac = 0.1  # test data ratio for autoencoder
    nr_auto_encode = int(test_frac * len(np_spec))
    idx_all = range(len(np_spec))
    idx_test = np.random.choice(idx_all, size=nr_auto_encode, replace=False)

    x_test = np_spec[idx_test]
    x_train = np_spec[np.delete(idx_all, idx_test, 0)]

    # normalize values
    norm_vec = np.linalg.norm(np_spec, axis=0)
    x_train = x_train / norm_vec
    x_test = x_test / norm_vec

    return x_train, x_test


def getModelName(layersizes, epochs, kenmerk):
    modelname = 'models/rulModel_' + kenmerk + '_'
    for size in layersizes:
        modelname += '{:d}_'.format(size)
    modelname += '1_ep{:d}.h5'.format(epochs)
    return modelname


def trainModel(X, Y, layersizes, epochs, kenmerk):
    input_dim = np.shape(X)[1]

    rulModel = Sequential()
    rulModel.add(Dense(layersizes[0], input_dim=input_dim, activation='relu', name='layer1'))
    for i in range(1, len(layersizes)):
        rulModel.add(Dense(layersizes[i], activation='relu', name='layer{:d}'.format(i + 1)))
    rulModel.add(Dense(1, activation='linear', name='output_layer'))
    rulModel.compile(loss='mean_squared_error', optimizer='adam')
    history = rulModel.fit(X, Y, epochs=epochs, shuffle=True, verbose=1)
    rulModel.save(getModelName(layersizes, epochs, kenmerk))

    plotTrainResults(X, Y, rulModel, history)

    return rulModel, history


def plotTrainResults(X, Y, rulModel, history):
    plt.plot(history.history['loss'])
    plt.show()

    plt.figure()
    predictions = rulModel.predict(X)
    plt.plot(predictions)
    plt.plot(Y)
    plt.show()


def evaluate_signal(modelnames, signal, RUL_value):
    spectro = getSpectro(signal)
    evaluate_spectro(modelnames, spectro, RUL_value)


def evaluate_spectro(modelnames, spectro, RUL_value, plotname=''):
    # True Y values
    nr_samples = len(spectro)
    Y_values = np.array([10 * (nr_samples - i) + RUL_value for i in range(1, nr_samples + 1)])

    models = [load_model(mname) for mname in modelnames]  # type: Model
    predictions = [(model.predict(spectro), np.sqrt(model.evaluate(spectro, Y_values))) for model in models]

    x_ax_values = [10 * i for i in range(nr_samples)]

    nr = 0
    for (prediction, rms) in predictions:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x_ax_values, prediction,label='Predicted RUL')

        plt.plot(x_ax_values, Y_values,label='Actual RUL')
        plt.title(modelnames[nr] + ' - RMS: {:.0f}'.format(rms))
        plt.ylabel('RUL (s)')
        plt.xlabel('Running Time (s)')
        plt.legend()

        if len(plotname) > 0:
            plt.savefig('plots/' + plotname, bbox_inches='tight', dpi=300)  # 600
        else:
            plt.show()

        nr += 1
