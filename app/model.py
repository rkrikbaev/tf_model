"""
    Class TensorFlow:


"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import keras
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import datetime

from utils import get_logger, LOG_LEVEL

logger = get_logger(__name__, loglevel=LOG_LEVEL)
logger.info(f'LOG_LEVEL: {LOG_LEVEL}')


class Model():

    def __init__(self):
        self.model = None

    def run(self, dataset, config, model_uri, period=None):

        print("\n**** mlflow.keras.load_model\n")
        logger.debug(model_uri)
        model = mlflow.keras.load_model(model_uri)
        #model = keras.models.load_model(model_uri, compile=False)
        #model.compile(optimizer='adam', loss='mean_absolute_error')
        logger.debug('Model loaded')
        X_series, _min, _max, scalers = self.prepare_dataset(dataset, column_index=0)

        input_window = config.get('input_window')
        output_window = config.get('output_window')
        granularity = config.get('granularity')

        assert X_series.shape[0] == input_window +1

        in_data = self.slice_data(X_series, input_window)
        result = model.predict(in_data)[0]
        result = np.reshape(result, (1,len(result)))
        result = scalers[0].inverse_transform(result)[0].tolist()
        values = list(map(lambda x: float(x), result))
        start_point = dataset[-1][0]
        
        if not isinstance(start_point, (int, float)): RuntimeError('start_point is not integer or float')
        series = self.to_series(values, int(start_point), granularity, output_window)

        return series

    def prepare_dataset( self, dataset, column_index ):
        
        logger.debug(f'Prepare dataset with length: {len(dataset)}')
        # Convert dataset to pandas DataFrame
        X = pd.DataFrame(dataset)

        X.set_index(X.columns[0], inplace=True)
        X['dt'] = pd.to_datetime(X.index)

        a = pd.DataFrame()
        anomalies = pd.DataFrame()
        threshold = 0.0
        f = 0
        for i in range(1, len(X)):
          if np.isnan(X.iloc[i,0]):
            anomalies = pd.concat([anomalies,X.iloc[[i]]])
            anomalies.iloc[-1, 0] = 0

          delta = abs(X.iloc[i, 0] - X.iloc[i-1, 0])
          if delta <= threshold:
            f = 1
            a = pd.concat([a, X.iloc[[i]]])
            if len(a) == 1:
              a = pd.concat([a, X.iloc[[i-1]]])
          elif delta > threshold and f == 1:
            f = 0
            if len(a) >= 3:
              anomalies = pd.concat([anomalies, a])
            a.drop(a.index, inplace=True)

        max_load = max(list(X.iloc[:,0]))
        min_load = min(list(X.iloc[:,0]))
        X = X.replace(np.nan, 2*min_load-max_load)
        for i in range(len(anomalies)):
          X.loc[anomalies.index[i]] = 2*min_load-max_load
          
        # create additional features from date
        # Day of week
        X['of_day'] = X['dt'].dt.dayofweek
        # of week
        X['of_week'] = X['dt'].dt.week
        # of month
        X['of_month'] = X['dt'].dt.month

        X.drop('dt', inplace=True, axis=1) 

        logger.debug(f'Normalize data {X.shape}')
        scalers = []

        for i in range(len(list(X.columns))):
          feature = np.reshape(list(X.iloc[:,i]), (len(X.iloc[:,i]), 1))
          
          if i in [0] and len(anomalies) > 0:
            scaler = MinMaxScaler(feature_range=(-1,1)).fit(feature)
          else:
            scaler = MinMaxScaler(feature_range=(0,1)).fit(feature)
          scalers.append(scaler)
          scaled_feature = np.reshape(scaler.transform(feature),len(X.iloc[:,i])).tolist()
          X.iloc[:,i] = scaled_feature
          
        _max = X[X.columns[column_index]].max()
        _min = X[X.columns[column_index]].min()
        X_series = np.array(X.values)
        
        return X_series, _min, _max, scalers

    def slice_data(self, X_series, input_window):
        logger.debug(f'Prepare matrix to slice data: {X_series.shape}')
        # create sclice
        N = X_series.shape[0]
        k = N - input_window
        X_slice = np.array([range(i, i + input_window) for i in range(k)])
        X_data = X_series[X_slice,:]

        in_data = X_data[0]
        in_data = np.reshape(in_data, (1, input_window, X_data.shape[2]))

        return in_data

    def to_series(self, values, start_point, granularity, output_window):

        base = datetime.datetime.fromtimestamp(start_point/1000 + granularity)
        date_list = [int((base + datetime.timedelta(hours=x)).timestamp()) for x in range(output_window)]
        series = [ list(x) for x in list(zip(date_list, values)) ]
        # logger.debug(f'Series: {series}')

        return series

