#First, we need to import the dataset from the Scikit-learn library, or else you can find structured datasets from platforms like Kaggle. But for now, we are using the Iris dataset prebuilt on Scikit-learn.
#1
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris() #Loading the dataset
iris.keys()
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

#2
# Converting the dataset to pandas dataframe Well, we have the data in our hands but it's not well structured for us to understand. So we need to convert it into a pandas DataFrame. Pandas is a great tool for doing all sorts of things related to datasets, including preprocessing and exploring them. So let's convert our dataset that is in the form of matrices into the form of rows and columns.
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )
#3
#Now we will be using Pandas' built-in function 'head()' to see the first few rows of our data frame.

print('Table with headings is \n', (iris.head(10)))

#4
#Here you can see that the iris data frame contains the length and width of sepals and petals including the target column which is the numerical representation of classes of Iris flowers that we need to classify (eg: Setosa(0), Versicolor(1),  Virginica(2) ).

#Since there is no column of names of species in the data frame let's add one more column with names of different species corresponding to their numerical values. It really helps us to access the different classes using their names instead of numbers.
species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species

#to group by species
print(iris.groupby('species').size())

#5
#You can also get some simple statistical information about the dataset by the "describe" method:

print('Table with descriptive statistics is \n',iris.describe())

#6
#to plot the data set in 2D
import matplotlib.pyplot as plt

setosa = iris[iris.species == "setosa"]
versicolor = iris[iris.species=='versicolor']
virginica = iris[iris.species=='virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

# lables and scatter points
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'], label="Versicolor", facecolor="green")
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red")


ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()
plt.show()

#7
#It’s pretty obvious to us humans that Iris-virginica has larger petals than Iris-versicolor and Iris-setosa. But computers cannot understand like we do. It needs some algorithm to do so. In order to achieve such a task, we need to implement an algorithm that is able to classify the iris flowers into their corresponding classes.
#Luckily we don't need to hardcode the algorithm for classification since there are already many algorithms available in the sci-kit learn package. We can simply choose any of them and use them. Here, I am going to use the Logistic Regression model. Now, after training our model on training data, we can predict petal measurements on testing data. And that's it! Before importing our Logistic model we need to convert our pandas' data frame into NumPy arrays. It is because we cannot apply the pandas data frame to an algorithm directly. Also, we can use the train_test_split function in sklearn in order to split the dataset into train and test,
from sklearn.model_selection import train_test_split

# Droping the target and species since we only need the measurements
X = iris.drop(['target','species'], axis=1)

# converting into numpy array and assigning petal length and petal width
X = X.to_numpy()[:, (2,3)]
y = iris['target']

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

#Alright! now we have all the stuff necessary for the Logistic Model, so let's import and train it.

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

#training_prediction
training_prediction = log_reg.predict(X_train)
print('Array with training preditcions is \n',training_prediction)

#Test predictions
test_prediction = log_reg.predict(X_test)
print('Array with testing preditcions is \n',test_prediction)

#8
#Performance measures are used to evaluate the effectiveness of classifiers on different datasets with different characteristics. For classification problems, there are three main measures for evaluating the model, the precision(the accuracy of positive predictions or the number of most relevant values from retrieved values.), Recall(ratio of positive instances that are truly detected by the classifier), and confusion matrix.

# Performance in training
from sklearn import metrics

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

#Another better way to evaluate the performance of a classifier is to look at the confusion matrix. The main usage of the confusion matrix is to identify how many of the classes are misclassified by the classifier.
print("Precision, Recall, Confusion matrix, in testing\n")


# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))
