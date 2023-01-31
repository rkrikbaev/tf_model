"""
    Class TensorFlow:


"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from pathlib import Path
import datetime
from sklearn.preprocessing import MinMaxScaler

from utils import get_logger, LOG_LEVEL

logger = get_logger(__name__, loglevel=LOG_LEVEL)
logger.info(f'LOG_LEVEL: {LOG_LEVEL}')


class Model():
    
    def __init__(self, tracking_server):
        self.model = None
        try:
            mlflow.set_tracking_uri(tracking_server)
        except:
            logger.error("""Couldn't connect to remote MLFLOW tracking server""")

    def run(self, dataset, config, model_uri, **kwargs):

        experiment_id = model_uri.get('experiment_id')
        run_id = model_uri.get('run_id')

        model_uri = f'/mlruns/{experiment_id}/{run_id}/mlmodel'

        print("\n**** mlflow.keras.load_model\n")
        model = mlflow.keras.load_model(model_uri)

        X_series, scalers = self.prepare_dataset(dataset, column_index=0)

        input_window = config.get('input_window')
        if not input_window: raise RuntimeError('input_window is empty')
        output_window = config.get('output_window')
        if not output_window: raise RuntimeError('output_window is empty')
        granularity = config.get('granularity')
        if not granularity: raise RuntimeError('granularity is empty')

        assert X_series.shape[0] == input_window +1

        in_data = self.slice_data(X_series, input_window)
        pred = model.predict(inp, verbose=0).tolist()[0]
        pred = np.reshape(pred, (1,len(p)))
        result = scalers[0].inverse_transform(p)[0]

        values = list(map(lambda x: float(x), result))
        start_point = dataset[-1][0]
        
        if not isinstance(start_point, (int, float)): RuntimeError('start_point is not integer or float')
        series = self.to_series(values, int(start_point), granularity, output_window)

        return series

    def prepare_dataset(self, dataset, column_index):
        
        logger.debug(f'Prepare dataset with length: {len(dataset)}')
        # Convert dataset to pandas DataFrame
        X = pd.DataFrame(dataset)
        
        anomalies = pd.DataFrame()

        for i in range(1, len(X)):
          if np.isnan(X.iloc[i,0]):
            anomalies = pd.concat([anomalies,X.iloc[[i]]])
            anomalies.iloc[-1, 0] = 0

        X['Time'] = pd.to_datetime(X['Time'])
        X.index = X['Time']
        X = X.drop([X.columns[0],X.columns[2]], axis=1)

        # create additional features from date
        X['Month'] = X.index.month
        X['Week'] = X.index.isocalendar().week.astype(np.int64)
        X['Day of week'] = X.index.dayofweek
        
        if len(anomalies) > 0:
            max_load = max(list(X.iloc[:,0]))
            min_load = min(list(X.iloc[:,0]))
            X = X.replace(np.nan, 2*min_load-max_load)
            for i in range(len(anomalies)):
              X.loc[anomalies.index[i]] = 2*min_load-max_load

        logger.debug(f'Normalize data {X.shape}')
        # Normalize features 
        scalers = []

        for i in range(len(list(X.columns))):
          feature = np.reshape(list(X.iloc[:,i]), (len(X.iloc[:,i]), 1))

          if i in [0]:
            scaler = MinMaxScaler(feature_range=(-1,1)).fit(feature)
          else:
            scaler = MinMaxScaler(feature_range=(0,1)).fit(feature)
          scalers.append(scaler)
          scaled_feature = np.reshape(scaler.transform(feature),len(X.iloc[:,i])).tolist()
          X.iloc[:,i] = scaled_feature
        X_series = X
        return X_series, scalers

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
