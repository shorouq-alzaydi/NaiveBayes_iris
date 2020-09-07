import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


data = load_iris()

print(data.DESCR)

# Explanatory variables
X = data['data']
columns = list(data['feature_names'])
print("Feature Names" + str(columns))

# Response variable
Y = data['target']
labels = list(data['target_names'])
print("Target Names" + str(labels))

# Visualize the frequency table
ser = pd.Series(Y)
table = ser.value_counts()
table = table.sort_index()
sns.barplot(labels,table.values)
plt.show()
#plotting.scatter

X_df = pd.DataFrame(X,columns=['Sepal_L','Sepal_W','Petal_L','Petal_W'])
my_cols_dict = {0:'red', 1:'green', 2:'blue'}
my_cols = pd.Series(Y).apply(lambda x: my_cols_dict[x])
pd.plotting.scatter_matrix(X_df, c=my_cols, marker='o', alpha=0.5)
plt.show()

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

GNB = GaussianNB()
GNB.fit(X_train,Y_train)
Y_pred_test = GNB.predict(X_test)

Confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred_test)
print(Confusion_matrix)

acc = metrics.accuracy_score(Y_test, Y_pred_test)
print('Accuracy    = ' + str(np.round(acc,3)))

#Visualize the posterior probabilities

centers = GNB.theta_
variances = GNB.sigma_
for i in range(4):
    x_min = X[:,i].min()
    x_max = X[:,i].max()
    x_range = x_max-x_min
    x_grid = np.linspace(x_min-x_range/3,x_max+x_range/3,300)
    fig=plt.figure(figsize=(4,2), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlabel(columns[i])
    ax.set_ylabel('Probability')
    for j in range(3):
        center = centers[j,i]
        sigma = np.sqrt(variances[j,i])
        ax.plot(x_grid, st.norm.pdf(x_grid,loc=center,scale=sigma),color=my_cols_dict[j],linestyle="--",label=labels[j])
    ax.legend(loc=0)
    plt.show()