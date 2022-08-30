from Entropy_Classifier import P1, P2
#from SPEED import Z1,Z2
from sklearn.datasets import make_circles
import pandas as pd
import numpy as np
from matplotlib.colors import  ListedColormap
import time
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import datasets
#"Divide os dados em treino e teste aleatoriamente" 
  
def mytrain_test_split(X,y, train_size): 
    X_train = X.sample(frac = train_size, random_state=2021)
    X_test = X.drop(X_train.index)
    y_train = y.sample(frac = train_size,random_state=2021)
    y_test = y.drop(y_train.index) 
    label = str(y_train.columns.values[0])
    y_test = y_test[label].values.tolist()
    return X_train, X_test, y_train, y_test


def plot_clf(model, y_test, X_test, activationname):
    if activationname == 'sigmoide':
        treshhold = 0.5
    if activationname == 'heaviside':
        treshhold = 0
    h = .01  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #X_test = X_test.to_numpy()
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    x_new = np.c_[xx.ravel(), yy.ravel()]
    print("te")
    #x_new = pd.DataFrame(x_new, columns=['x1','x2'])
    Z = model.predict(x_new,treshhold)
    print("d")
    Z =  np.array( Z)
    Z = Z.reshape(xx.shape)
    # Put the result into a color plot
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification")
    return fig

def plot_clf_sk(model, y_test, X_test):
    h = .01  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #X_test = X_test.to_numpy()
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    x_new = np.c_[xx.ravel(), yy.ravel()]
    #x_new = pd.DataFrame(x_new, columns=['x1','x2'])
    Z = model.predict(x_new)
    print("d")
    Z =  np.array( Z)
    Z = Z.reshape(xx.shape)
    # Put the result into a color plot
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification")
    return fig


""" Running from Desktop 

    #-----------------------------Data Preparation----------#
tran_val = 70/100
X, y=make_circles(n_samples=200,noise=0.01, factor= 0.2, random_state=(2021))
X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=['classe'])
X_train, X_test, y_train, y_test = mytrain_test_split(X,y, tran_val)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
"""
"""
dataset = pd.read_excel("C:\\Users\\PC\\Documents\\Material Tese\\Datasets Benshmark\\teste_linear_espelhado.xlsx")

dataset = dataset.dropna(axis=1, how='all')
dataset = dataset.drop_duplicates()
#Comment if use Sklearn array datasets
X = dataset.drop(['classe'], axis=1)
y = dataset['classe'].to_frame()
X_train = X.to_numpy()
y_train = y.to_numpy()
arr_y = np.where(y_train == 0)
y_train[arr_y] = 2
y_test = y["classe"].values.tolist()
"""
dataset = pd.read_csv("C:\\Users\\PC\\Documents\\Material Tese\\Datasets Benshmark\\dataR2.csv")

dataset = dataset.dropna(axis=1, how='all')
dataset = dataset.drop_duplicates()
#Comment if use Sklearn array datasets
X = dataset.drop(['classe'], axis=1)
y = dataset['classe'].to_frame()
X_train = X.to_numpy()
y_train = y.to_numpy()
arr_y = np.where(y_train == 0)
y_train[arr_y] = 2
y_test = y["classe"].values.tolist()

#digits = load_digits()
#X = digits.data
#y = digits.target
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
#y_train = np.reshape(y_train,(-1,1)) #Change the (lines,) to (lines,1) (1d array in n numpy doesn't have x2 axis)
"""
iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
y_train = y_train.reshape(-1,1)
"""
#X_train = X_train.to_numpy()

#
print("teste")
""" Running from Desktop 
     #----------------------------Data Preparation-------#
start_time = time.time()
model1 = P1(activation='heaviside')
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
len(np.unique(y_train))
folhas = [0.01]
Entropia = [10**-3,10**-5]
galhos = [10]
best_params_calc,list_S, list_r,acc ,pre, rec, cmd, grid = model1.grid_search(X_train, y_train, X_test, y_test, Entropia, folhas, galhos)
#print('Positive Frequency =', grid[0][0]['f_pos'])
#print('Negative Frequency =', grid[0][0]['f_neg'])
plot_clf(model1, y_train, X_train, activationname="heaviside")
print("First One--- %s seconds ---" % (time.time() - start_time))
"""


"""

print("Beginning Z1 Classification")
start_time3 = time.time()
model2 = Z2(activation='sigmoide')
#X_train = X_train.to_numpy()

#X_test = X_test.to_numpy()
#y_train = y_train.to_numpy()
#len(np.unique(y_train))
folhas = [0.01]
Entropia = [10**-3]
galhos = [10]
best_params_calc,list_S, list_r,acc ,pre, rec, cmd, grid = model2.grid_search(X_train, y_train, X_train, y_test, Entropia, folhas, galhos)
#print('Positive Frequency =', grid[0][0]['f_pos'])
#print('Negative Frequency =', grid[0][0]['f_neg'])
plot_clf(model2, y_train, X_train, activationname="sigmoide")
print("First One--- %s seconds ---" % (time.time() - start_time3))
"""

print("Beginning P1 Classification")
start_time = time.time()
model1 = P2(activation='sigmoide')
#X_train = X_train.to_numpy()
#X_test = X_test.to_numpy()
#y_train = y_train.to_numpy()
#len(np.unique(y_train))
folhas = [0.01]
Entropia = [10**-3]
galhos = [50]
#best_params_calc,list_S, list_r,acc ,pre, rec, cmd, grid = model1.grid_search(X_train, y_train, X_test, y_test, Entropia, folhas, galhos)
#using my dataset
best_params_calc,list_S, list_r,acc ,pre, rec, cmd, grid = model1.grid_search(X_train, y_train, X_train, y_train, Entropia, folhas, galhos)

#print('Positive Frequency =', grid[0][0]['f_pos'])
#print('Negative Frequency =', grid[0][0]['f_neg'])
#plot_clf(model1, y_train, X_train, activationname='sigmoide')
print("Second One--- %s seconds ---" % (time.time() - start_time))

#print("Beginning DecisionTree Classification")
#start_time2 = time.time()
#clf = DecisionTreeClassifier()
#clf = clf.fit(X_train, y_train) 
#plot_clf_sk(clf, y_train, X_train)
#print(" Third Time One--- %s seconds ---" % (time.time() - start_time2))
"""
Created on Sat Aug 13 15:48:12 2022

@author: PC


def test( *kargs,**args ):
    print(args)
    print(kargs)
    print(args[0])
    print(kargs.get('a'))

alpha = 'alpha'
beta = 'beta'
test(alpha, beta, a=1, b=2)


def team(name, project, members=None):
    team.name= name
    team.project= project
    team.members= members
    print(name, "is working on an", project)
    
team("Edpresso", project = "FemCode")

teste = "lucas"
lista = [1,2]
result = zip(teste,lista)
print(list(result))

lista2 = [1,2,3,4,5,6,7]

for index, element in enumerate(lista2):
    print(element)
    #print(element)

"""






