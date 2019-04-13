import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn import svm, datasets
import sys

sys.stdout = open('results.txt','wt') 


def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("Confusion Matrix for Bank Data")
    print(confusion_matrix(y_test,y_pred))  
    print("Classification Report for Bank Data")
    print(classification_report(y_test,y_pred))  


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 
    
    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # X_arr = X.values  # covert to ndarray
    X_arr = X[:, :2]  # we only take the Sepal two features for Plotting.
    y_arr = y.replace({'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2 }) # convert y val to 0, 1, 2
    y_arr = y_arr.values
    # train
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 
    return (X_arr,y_arr,X_train, X_test, y_train, y_test)

def polynomial_kernel(X,y,X_train, X_test, y_train, y_test):
    # TODO
    # NOTE: use 8-degree in the degree hyperparameter. 
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='poly',degree=8 ,gamma='auto')  
    model = svclassifier
    plot_model(model,'SVC with polynomial (degree 8) kernel',X,y)
    
    # Fit and predict
    svclassifier.fit(X_train, y_train)  
    
    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("Confusion Matrix for Polynomial")
    print(confusion_matrix(y_test,y_pred))
    print("Classification report for Polynomial")
    print(classification_report(y_test,y_pred))
    print("Score for polynomial SVM:",svclassifier.score(X_train, y_train))
    
    
    
    
def gaussian_kernel(X,y,X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF  
    kernel = 1.0 * RBF(1.0)
    svclassifier = GaussianProcessClassifier(kernel=kernel,random_state=0) 
    model = svclassifier
    plot_model(model,'Gaussian kernel',X,y)
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("Confusion Matrix for gaussian")
    print(confusion_matrix(y_test,y_pred))
    print("Classification report for gaussian")
    print(classification_report(y_test,y_pred))
    print("Score for Gaussian RBF:",svclassifier.score(X_train, y_train)) 
    # Trains, predicts and evaluates the model

def sigmoid_kernel(X,y,X_train, X_test, y_train, y_test):
    # TODO
    
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='sigmoid', gamma='auto')  
  
    model = svclassifier
    plot_model(model,'Sigmoid kernel',X,y)
    svclassifier.fit(X_train, y_train)  
    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("Confusion Matrix for sigmoid")
    print(confusion_matrix(y_test,y_pred))
    print("Classification report for sigmoid")
    print(classification_report(y_test,y_pred))
    print("Score for Sigmoid SVM:", svclassifier.score(X_train, y_train))
    # Trains, predicts and evaluates the model
    

def plot_model(model,title,X,y):
    
    model = model.fit(X, y)
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
 
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()
    return

def test():
    linear_svm()
    X,y,X_train, X_test, y_train, y_test = import_iris()
    polynomial_kernel(X,y,X_train, X_test, y_train, y_test)
    gaussian_kernel(X,y,X_train, X_test, y_train, y_test)
    sigmoid_kernel(X,y,X_train, X_test, y_train, y_test)
 

test()
