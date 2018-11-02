import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

header_names = ['Date',
 'Open',
 'High',
 'Low',
 'Close',
 'Volume',
 'Dividend',
 'Split Ratio',
 'Adj. Open',
 'Adj. High',
 'Adj. Low',
 'Adj. Close',
 'Adj. Volume']


df = pd.read_csv('EOD-KO.csv', header=None, names=header_names,  error_bad_lines = False)

df[df['Open'] == 0]

#df.loc[:,'Daily Variation'] = df.loc[:,'High'] - df.loc[:,'Low']
#df.loc[:,'Percentage Variation'] = df.loc[:,'Daily Variation'] / df.loc[:,'Open'] * 100
#df.loc[:,'Adj. Daily Variation'] = df.loc[:,'Adj. High'] - df.loc[:,'Adj. Low']
#df.loc[:,'Adj. Percentage Variation'] = df.loc[:,'Adj. Daily Variation'] / df.loc[:,'Adj. Open'] * 100

def prepare_train_test(days, periods, target='Adj. Close', test_size=0.2, buffer=0, target_days=7):  
    """Returns X_train, X_test, y_train, y_test for parameters.
    Predicts prices `target_days` ahead.
    `days` = number of days prior we consider"""
    # Columns
    columns = []
    for j in range(1,days+1):
        columns.append('i-%s' % str(j))
    columns.append('Adj. High')
    columns.append('Adj. Low')

    # Columns: Prices (predict multiple day)
    nday_columns = []
    for j in range(1,target_days+1):
        nday_columns.append('Day %s' % str(j-1))

    # Index
    start_date = df.iloc[days+buffer]["Date"]
    print (start_date)
    index = pd.date_range(start_date, periods=periods, freq='D')

    # Create empty dataframes for features and prices
    features = pd.DataFrame(index=index, columns=columns)
    prices = pd.DataFrame(index=index, columns=["Target"])
    nday_prices = pd.DataFrame(index=index, columns=nday_columns)

    # Prepare test and training sets
    for i in range(periods):
        # Fill in Target df
        for j in range(target_days):
            nday_prices.iloc[i]['Day %s' % str(j)] = df.iloc[buffer+i+days+j][target]
        # Fill in Features df
        for j in range(days):
            features.iloc[i]['i-%s' % str(days-j)] = df.iloc[buffer+i+j][target]
        features.iloc[i]['Adj. High'] = max(df[buffer+i:buffer+i+days]['Adj. High'])
        features.iloc[i]['Adj. Low'] = min(df[buffer+i:buffer+i+days]['Adj. Low'])
                
    X = features
    y = nday_prices
    #print("X.tail: ", X.tail())

    # Train-test split
    if len(X) != len(y):
        return "Error"
    split_index = int(len(X) * (1-test_size))
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test


X_train = []
X_test = []
y_train = []
y_test = []


from sklearn.multioutput import MultiOutputRegressor

# Import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error


def rmsp(test, pred):
    return np.sqrt(np.mean(((test - pred)/test)**2)) * 100

def print_metrics(test, pred):
    print("Root Mean Squared Percentage Error", rmsp(test, pred))
    print("Mean Absolute Error: ", mean_absolute_error(test, pred))
    print("Explained Variance Score: ", explained_variance_score(test, pred))
    print("Mean Squared Error: ", mean_squared_error(test, pred))
    print("R2 score: ", r2_score(test, pred))



from sklearn import svm
from sklearn.linear_model import LinearRegression


# In[24]:


# Initialise variables to prevent errors
days = 7


def classify_and_metrics(clf=LinearRegression(), target_days=7, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, days=days):
    """Trains and tests classifier on training and test datasets.
    Prints performance metrics.
    """
    # Classify and predict
    clf = MultiOutputRegressor(clf)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # Lines below for debugging purposes
#    print("X_train.head(): ", X_train.head())
#    print("X_train.tail(): ", X_train.tail())
#    print("Pred: ", pred[:5])
#    print("Test: ", y_test[:5])
    
    # Print metrics
    print("# Days used to predict: %s" % str(days))
    print("\n%s-day predictions" % str(target_days)) 
    print_metrics(y_test, pred)
    return rmsp(y_test, pred)


def execute(steps=8, buffer_step=1000, days=7, periods=1000, model=LinearRegression(), predict_days=7):
    """Performs `steps` train-test cycles and prints evaluation metrics for BP data.
    `steps`: number of train-test cycles.
    `periods`: the total number of datapoints used in each cycle (training + test)
    `buffer_step`: number of datapoints between the starting points of each
    consecutive train-test cycle
    """
    errors=[]
    r2=[]
    for segment in range(steps):
        buffer = segment*buffer_step
        print("Buffer: ", buffer)
        X_train, X_test, y_train, y_test = prepare_train_test(days=days, periods=periods, buffer=buffer)
        errors.append(classify_and_metrics(clf=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, days=days))
    print("Errors: ", errors)
    
    daily_error = []
    for target_day in range(predict_days):
        daily_error.append([])
    for segment in range(steps):
        for target_day in range(predict_days):
            daily_error[target_day].append(errors[segment][target_day])
    print("Daily error: ", daily_error)
    average_daily_error = []
    for day in daily_error:
        average_daily_error.append(np.mean(day))
    print("Mean daily error: ", average_daily_error)


execute(steps=10,days=7, buffer_step = 700)
#execute(model=svm.SVR(), steps=8)
#execute(steps=10, days=10, buffer_step = 700)


