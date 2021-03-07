import keras
import keras.wrappers.scikit_learn
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import re
import tsensor

try:
    def build_net():
        model = keras.models.Sequential([keras.layers.Dense(units=1, input_dim=2)])
        model.compile(loss=keras.losses.mean_squared_error, optimizer="sgd")
        return model

    regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_net)
    # Works with the sklearn regressors
    # regressor = sklearn.linear_model.LinearRegression()
    X = np.zeros((1, 2))
    Y = np.zeros((1,))
    regressor.fit(X, Y)
    Y_pred = regressor.predict(X)
    print(Y_pred.shape)  # Is (), should be (1,)
    # As a result, this fails with an exception
    # TypeError: Singleton array array(0., dtype=float32) cannot be considered a valid collection.
    with tsensor.clarify():
        print(sklearn.metrics.mean_squared_error(y_true=Y, y_pred=Y_pred))
except Exception as e:
    print(e)
