import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import namedtuple

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def detect_outlier_with_regression():
    df = pd.read_csv(get_fullpath('accord_sedan_training.csv'))
    # df.columns
    # print df.trim.value_counts()
    # print df.year.median()

    le = LabelEncoder()
    df['trim_encode'] = le.fit_transform(df.trim)
    df['engine_encode'] = le.fit_transform(df.engine)
    df['transmission_encode'] = le.fit_transform(df.transmission)
    # print df[['price', 'mileage', 'trim_encode', 'engine_encode', 'transmission_encode']].corr()
    X = df[['mileage', 'trim_encode', 'engine_encode', 'transmission_encode']]
    y = df.price

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    # lr.coef_

    result = pd.DataFrame({'predict':lr.predict(X)})
    result['error'] = np.abs(lr.predict(X) - y)
    result.describe()
    outlierIdx = result.error >= np.percentile(result.error, 90)

    # 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(df.mileage, df.trim_encode, df.price, c='b')
    ax.scatter(df.mileage[outlierIdx], df.trim_encode[outlierIdx], df.price[outlierIdx], c='r')
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Transmission')
    ax.set_zlabel('Price')
    # 2D
    # scatter(df.mileage, df.price, c=(0,0,1), marker='s')
    # scatter(df.mileage[outlierIdx], df.price[outlierIdx], c=(1,0,0), marker='s')plt.show()
    plt.show()

def detect_outlier_with_dbscan():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    df = pd.read_csv(get_fullpath('accord_sedan_training.csv'))
    le = LabelEncoder()
    df['trim_encode'] = le.fit_transform(df.trim)
    df['engine_encode'] = le.fit_transform(df.engine)
    df['transmission_encode'] = le.fit_transform(df.transmission)

    ss = StandardScaler()
    df.price = ss.fit_transform(df.price)
    df.mileage = ss.fit_transform(df.mileage)
    df.trim_encode = ss.fit_transform(df.trim_encode)
    df.engine_encode = ss.fit_transform(df.engine_encode)
    df.transmission_encode = ss.fit_transform(df.transmission_encode)

    X = df[['mileage', 'price', 'trim_encode', 'engine_encode', 'transmission_encode']]

    Param = namedtuple('Param', ['eps', 'min_samples'])
    param = Param(1.20, 5)
    dbscan = DBSCAN(eps=param.eps, min_samples=param.min_samples).fit(X.values)
    labels = dbscan.labels_
    unique_labels = set(labels)
    for k in unique_labels:
        for index in np.argwhere(labels == k):
#             print k, index
            col = 'r' if k == -1 else 'b'
            ax.scatter(df.mileage[index], df.trim_encode[index], df.price[index], c=col)
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Transmission')
    ax.set_zlabel('Price')
    plt.show()

if __name__ == '__main__':
#     detect_outlier_with_regression()
    detect_outlier_with_dbscan()