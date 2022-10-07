import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_plot(model,X_train, Y_train):# create plot
    fig, ax = plt.subplots()

    # get colors from qualitative colormap 'Paired'
    cmap = plt.cm.get_cmap('Paired')

    # plot data points
    ax.scatter(X_train.iloc[Y_train == 1, 0], X_train.iloc[Y_train == 1, 1],
               c=[cmap(11)], label='1')
    ax.scatter(X_train.iloc[Y_train == 0, 0], X_train.iloc[Y_train == 0, 1],
               c=[cmap(0)], label='0')
    ax.legend(loc='best')

    # plot the decision function
    # create grid to evaluate model
    x1_min, x1_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    x2_min, x2_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1

    XX, YY = np.meshgrid(np.arange(x1_min, x1_max, .2),
                         np.arange(x2_min, x2_max, .2))

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Establish the class for each point in the contour
    Z = model.predict(xy).reshape(XX.shape)

    # Visualization of the contour
    ax.contourf(XX, YY, Z, cmap='bwr', alpha=0.3)

    # plot support vectors, whose are responsible for building the margins
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', marker='s')

    ax.axis([x1_min, x1_max, x2_min, x2_max])
    plt.axis('tight')
    return plt


def compute_accuracy(y_true, y_pred):
    corrpredicts = 0
    for true, predicted in zip(y_true, y_pred):
        if true == predicted:
            corrpredicts += 1
    accuracy = corrpredicts/len(y_true)
    return accuracy


def compute_precision(y_test,y_pred):

    df = pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
    crosstab = pd.crosstab(df.y_test, df.y_pred)
    TP = crosstab[1][1]
    FP = crosstab[1][0]
    TN = crosstab[0][0]
    FN = crosstab[0][1]
    precision_1 = TP / (TP+FP)
    precision_0 = TN / (TN+FN)
    
    return precision_1, precision_0