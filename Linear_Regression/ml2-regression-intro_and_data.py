import pandas as pd
import quandl
import math
import datetime
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# my quandl token for access
quandl.ApiConfig.api_key = 'RxMnZ-vTo2gLHypn4pSb'
# Print Option for numpy
numpy.set_printoptions(threshold=numpy.inf)

# df print options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)

df = quandl.get('WIKI/GOOGL')  # receiving data


df = df[['Adj. Open', 'Adj. High', 'Adj. Low',
         'Adj. Close', 'Adj. Volume', ]]  # resorting dataframe
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / \
    df['Adj. Close'] * 100.0  # High-Low Percentage at end of day
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / \
    df['Adj. Open'] * 100.0  # High-Low Percentage at the following day

# adjusted dataframe for further action
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# print(df.head())

forecast_col = 'Adj. Close'  # We will try to forcast this column
df.fillna(-99999, inplace=True)  # clearing rows with Nan

# taking a fix number from length of db to predict
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)  # the number

# adding a column name label where all data will
# be shited with the forecast_out number
df['label'] = df[forecast_col].shift(-forecast_out)

# Normal data set without label in column and converted to array for operation
X = np.array(df.drop(['label'], 1))

# Standerdising the data
X = preprocessing.scale(X)
# print(X)

# Slicig the array to a limit
X = X[:-forecast_out]
# Sliced arrat after limit which we will try to predict
X_lately = X[-forecast_out:]

df.dropna(inplace=True)  # again we are clearing row without Nan
y = np.array(df['label'])  # the result table we want to test with


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)  # Spliting the array in order to train and test


# object(classifier) for Linear_Regression Model, and storing
# several important data like coef...
# clf = LinearRegression(n_jobs=-1)
#
# # fit data into model
# clf.fit(X_train, y_train)
#
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)  # accuracy percentage of classifire
# print (accuracy)

# this variable contains data which is forcasted from model against X_lately
forecast_set = clf.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)

# created new column with Nan value
df['forecast'] = np.nan


last_date = df.iloc[-1].name  # date of last data from dataframe
last_unix = last_date.timestamp()  # converting to timestamp(number in seconds)
one_day = 86400  # variable to add one day
# So this variable now have the next date of last data in df(dataframe)
next_unix = last_unix + one_day


# extending df with forcast data while whole df will have both data
# but where is forrecast set data, there will be all other coloumns Nan
for i in forecast_set:
    # contains the last date received from next unix
    next_date = datetime.datetime.fromtimestamp(
        next_unix)  # convert to datetime
    next_unix += one_day
    # location of the date will be populated with Nan and last column will
    # be populated with forcast data
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
