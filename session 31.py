import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10000000

data = pd.read_csv('csv files/bank-additional-full.csv',sep=';')

def y(var):
    if var == 'yes':
        return 1
    elif var == 'no':
        return 2
    elif var == 'unknown':
        return 0
def education(var):
    if var == 'basic.4y':
        return 1
    elif var == 'basic.6y':
        return 2
    elif var == 'basic.9y':
        return 3
    elif var == 'high.school':
        return 4
    elif var == 'professional.course':
        return 5
    elif var == 'university.degree':
        return 6
    else:
        return 0
def marital(var):
    if var == 'married':
        return 1
    elif var == 'single':
        return 2
    elif var == 'divorced':
        return 3
    else:return 0
def contact(var):
    if var == 'cellular':
        return 0
    elif var == 'telephone':
        return 1
def pout(var):
    if var == 'nonexistent':
        return 0
    elif var == 'success':
        return 1
    elif var == 'failure':
        return 2
def day(var):
    if var == 'mon':
        return 1
    elif var == 'tue':
        return 2
    elif var == 'wed':
        return 3
    elif var == 'thu':
        return 4
    elif var == 'fri':
        return 5
def month(var):
    if var == 'jan':
        return 1
    elif var == 'feb':
        return 2
    elif var == 'mar':
        return 3
    elif var == 'apr':
        return 4
    elif var == 'may':
        return 5
    elif var == 'jun':
        return 6
    elif var == 'jul':
        return 7
    elif var == 'aug':
        return 8
    elif var == 'sep':
        return 9
    elif var == 'oct':
        return 10
    elif var == 'nov':
        return 11
    elif var == 'dec':
        return 12
def job(var):
    if var == 'admin.':
        return 1
    elif var == 'services':
        return 2
    elif var == 'retired':
        return 3
    elif var == 'management':
        return 4
    elif var == 'technician':
        return 5
    elif var == 'blue-collar':
        return 6
    elif var == 'housemaid':
        return 7
    elif var == 'student':
        return 8
    elif var == 'entrepreneur':
        return 9
    elif var == 'self-employed':
        return 10
    elif var == 'nov':
        return 11
    else:
        return 0

data['month'] = data['month'].apply(month)
data['day_of_week'] = data['day_of_week'].apply(day)
data['y'] = data['y'].apply(y)
data['default'] = data['default'].apply(y)
data['loan'] = data['loan'].apply(y)
data['housing'] = data['housing'].apply(y)
data['contact'] = data['contact'].apply(contact)
data['poutcome'] = data['poutcome'].apply(pout)
data['education'] = data['education'].apply(education)
data['marital'] = data['marital'].apply(marital)
data['job'] = data['job'].apply(job)

x = data['duration'].values.reshape(-1,1)
y = data['age'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
lr = LinearRegression()
lr.fit(x_train_scaled,y_train)
y_pred = lr.predict(x_test_scaled)

plt.scatter(x_test_scaled,y_test)
plt.plot(x_test_scaled,y_pred)
plt.show()

y_pred_round = np.round(y_pred)
print('mean squared error:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
print('regular accuracy:',metrics.accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('f1:',metrics.f1_score(y_true=y_test,y_pred=y_pred_round,average='weighted')*100,'%')
print('precision:',metrics.precision_score(y_true=y_test,y_pred=y_pred_round,average='weighted',zero_division=1)*100,'%')
print('kappa:',metrics.cohen_kappa_score(y1=y_test,y2=y_pred_round)*100,'%')

matrix = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred_round)
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()
