import numpy as np
import math

# sigmoid(logit(x)) = x
def logit(x):
    return np.log(x/(1-x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logLikelihood(X, y, beta):
    n = X.shape[0]
    sum = 0
    for i in range(n):
        xi = X[i, :]
        yi = y[i]
        z = sigmoid(np.dot(xi, beta))
        sum += yi * np.log(z) + (1 - yi) * np.log(1 - z)
    return sum

def gradientDescent(X, y, beta, learningRate, iterations):
    n = X.shape[0]
    for iteration in range(iterations):
        gradient = np.zeros(beta.shape)
        for i in range(n):
            xi = X[i, :]
            yi = y[i]
            z = sigmoid(np.dot(xi, beta))
            gradient += (z - yi) * xi
        beta = beta - learningRate * gradient
        # print("Value of logLikelihood : ", logLikelihood(X, y, beta), " with beta : ", beta)
    return beta

def addOnes(X):
    m, n = X.shape
    Xplus = np.zeros((m, n+1))
    Xplus[:, 0] = 1
    Xplus[:, 1:] = X
    return Xplus

def showDataset(X, y):
    for i in range(len(X)):
        print(X[i], y[i])

def sliceDataset(X, y):
    train = len(X)//5
    Xtrain = X[:train]
    ytrain = y[:train]
    Xtest = X[train:]
    ytest = y[train:]
    return Xtrain, ytrain, Xtest, ytest

def accuracy(ypred, ytest):
    l = len(ypred)
    s = 0
    for i in range(l):
        if ypred[i] == ytest[i]:
            s = s + 1
    return s*100/l

def precision(ypred, ytest):
    l = len(ypred)
    tp = 0
    fp = 0
    for i in range(l):
        if ypred[i] == 1:
            if ypred[i] == ytest[i]: tp = tp + 1
            else: fp = fp + 1
    return (tp/(tp+fp))*100

def recall(ypred, ytest):
    l = len(ypred)
    tp = 0
    fn = 0
    for i in range(l):
        if ytest[i] == 1:
            if ypred[i] == ytest[i]: tp = tp + 1
            else: fn = fn + 1
    return (tp/(tp+fn))*100

def countOnes(y):
    s = 0
    for i in range(len(y)):
        if y[i] == 1: s = s + 1
    return s

def kFoldCrossValidation(X, y, beta, learningRate, iterations, k):
    print("Starting k-fold cross validation with parameters :")
    print("Number of subpart :", k)
    print("Starting beta :", beta)
    print("Learning rate :", learningRate)
    print("Iterations per gradient descent :", iterations)
    print("---------------")
    n = X.shape[0]
    foldSize = n//k
    metrics = []
    meanBeta = np.zeros(X.shape[1])
    for i in range(k):
        Xtest = X[i*foldSize:(i+1)*foldSize, :]
        ytest = y[i*foldSize:(i+1)*foldSize]
        Xtrain = np.concatenate((X[:i*foldSize, :], X[(i+1)*foldSize:, :]), axis=0)
        ytrain = np.concatenate((y[:i*foldSize], y[(i+1)*foldSize:]), axis=0)
        beta = gradientDescent(Xtrain, ytrain, beta, learningRate, iterations)

        ypred = []
        for xi in Xtest:
            if sigmoid(np.dot(xi, beta)) > 0.5: ypred.append(1)
            else: ypred.append(0)
        
        acc = accuracy(ypred, ytest.tolist())
        pre = precision(ypred, ytest.tolist())
        rec = recall(ypred, ytest.tolist())
        print("Using", i+1, "/", k, "part to test - Accuracy rate :", acc, "- Precision rate:", pre, "- Recall :", rec)
        print("with beta :", beta)
        metrics.append([acc, pre, rec])

        meanBeta = meanBeta + beta
    print("---------------")
    return metrics, meanBeta/k

def testPrediction(Xtest, y, testBeta):
    ypred = []
    for xi in Xtest:
        if sigmoid(np.dot(xi, testBeta)) > 0.5: ypred.append(1)
        else: ypred.append(0)
    
    acc = accuracy(ypred, y.tolist())
    pre = precision(ypred, y.tolist())
    rec = recall(ypred, y.tolist())
    print("Accuracy rate :", acc, "- Precision rate:", pre, "- Recall :", rec)
    print("with beta :", testBeta)
    return [acc, pre, rec]