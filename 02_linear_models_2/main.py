import numpy as np
import matplotlib as mpl
import pandas as pd
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


def sample_2d_gaussian(meanx, meany, variance_x, variance_y, covariance, numsamps):
    '''
    Generates a random sample of size 'numsamps' from a 2-dimensional Gaussian distribution.
    The Gaussian is defined by the mean vector (meanx,meany) and the
    covariance matrix

    variance_x    covariance
    covariance    variance_y

    All parameters can be freely chosen, except covariance, which must satisfy the inequality

    covariance <= sqrt(variance_x * variance_y)
    '''
    meanvec = np.array([meanx, meany])
    covarmatrix = np.array([[variance_x, covariance], [covariance, variance_y]])
    return multivariate_normal(meanvec, covarmatrix, numsamps)


def maxpos(A):
    '''
    Takes an n x k array A, and returns 1-dim n array where the i'th
    entry is the index of column in A where the i'th row of A has its
    maximal value (application: turns a probabilitiy distribution over
    k classes for n instances into a single prediction)
    '''
    return np.argmax(A, axis=1)


# sampling data_depr from 2-dimensional Gaussian distributions. For reproducibility, one can fix a random seed.
def gauss_sampling(gauss_samples):
    # np.random.seed(1)
    datasize = 250
    mixturecoeff = np.array([0.4, 0.2, 0.4])
    componentsizes = (datasize * mixturecoeff).astype(int)

    class0samp = sample_2d_gaussian(gauss_samples[0][0], gauss_samples[0][1], gauss_samples[0][2], gauss_samples[0][3],
                                    gauss_samples[0][4], componentsizes[0])
    class1samp = sample_2d_gaussian(gauss_samples[1][0], gauss_samples[1][1], gauss_samples[1][2], gauss_samples[1][3],
                                    gauss_samples[1][4], componentsizes[1])
    class2samp = sample_2d_gaussian(gauss_samples[2][0], gauss_samples[2][1], gauss_samples[2][2], gauss_samples[2][3],
                                    gauss_samples[2][4], componentsizes[2])

    features = np.concatenate((class0samp, class1samp, class2samp), axis=0)
    labels = np.concatenate((np.zeros(componentsizes[0]), np.ones(componentsizes[1]), 2 * np.ones(componentsizes[2])))
    return features, labels


#  Loading mi.txt data_depr:
def mi_data():
    midata = pd.read_csv("MI-labeled.txt", sep=',')
    features = np.array(midata[['X1', 'X2']])
    classlabels = midata['Class']
    labels = np.zeros(len(classlabels))
    for i in range(len(classlabels)):
        if classlabels[i] == 'I':
            labels[i] = 1
    return features, labels


def part1(model, data, modelchoice, features, labels, title):
    # Preparing meshgrid for plotting decision regions:
    maxvalx = np.max(features[:, 0])
    maxvaly = np.max(features[:, 1])
    minvalx = np.min(features[:, 0])
    minvaly = np.min(features[:, 1])
    border = 2
    xinterval = (maxvalx - minvalx) / border
    yinterval = (maxvaly - minvaly) / border
    xx, yy = np.meshgrid(np.arange(minvalx - xinterval, maxvalx + xinterval, xinterval / 100),
                         np.arange(minvaly - yinterval, maxvaly + yinterval, yinterval / 100))

    # Applying model to the meshgrid. All models return a quantitative "likelihood" for the different classes.
    # For the probabilistic models, these are class label probabilities that are retrieved using the .predict_proba method.
    # For the non-probabilistic SVC model, this is the decision_function method. In all cases, we classify a datapoint as belonging to the class with the maximal "likelihood" value.
    if modelchoice == "svc":
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        if data == "mi":
            meshclasses = np.ones(Z.size)
            meshclasses[Z < 0] = 0
            meshclasses = meshclasses.reshape(xx.shape)
        else:
            meshclasses = maxpos(Z).reshape(xx.shape)
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        meshclasses = maxpos(Z).reshape(xx.shape)

    # Plotting datapoints and decision regions
    plt.title("Data: " + data + " Classifier: " + modelchoice + " set: " + title)
    plt.contourf(xx, yy, meshclasses, [-0.1, 0.1, 1, 2], colors=('tomato', 'lightblue', 'lightgreen'))
    plt.scatter(features[:, 0], features[:, 1], c=labels,
                cmap=mpl.colors.ListedColormap(['r', 'b', 'g']))
    # plt.scatter(data_depr[:,0],data_depr[:,1],c=classlabels_numeric, cmap = mpl.colors.ListedColormap(['r', 'b']))
    plt.show()

    # We also calculate the predictions on the (training) datapoints, and check the accuracy:
    pred_labels = model.predict(features)
    accuracy = accuracy_score(labels, pred_labels)
    print(f"Accuracy: {accuracy}")

    return accuracy


# Part 2
def part2(modelchoice):
    print("")
    bostondata = load_boston()

    # We generate our labels, and create a train/test split:
    labels = np.array([1 if y > np.median(bostondata['target']) else 0 for y in bostondata['target']])
    features = bostondata['data_depr']
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

    # For this investigation it may be helpful to normalize the data_depr before building the classifiers, using the code in the cell below (why is this useful?).
    # If i didnt normalize, i got some warning in gauss fit
    scaler = StandardScaler()
    scaler.fit(features_train)
    features_train_norm = scaler.transform(features_train)
    features_test_norm = scaler.transform(features_test)

    # Selecting a model class, learning, and measuring accuracy:
    if modelchoice == "gauss":
        model = GaussianNB()
    if modelchoice == "lda":
        model = LinearDiscriminantAnalysis()
    if modelchoice == "log":
        model = LogisticRegression()
    if modelchoice == "svc":
        model = SVC(kernel='linear')

    model.fit(features_train_norm, labels_train)
    pred_labels_train = model.predict(features_train_norm)
    pred_labels_test = model.predict(features_test_norm)

    train_score = accuracy_score(labels_train, pred_labels_train)
    test_score = accuracy_score(labels_test, pred_labels_test)

    print(f"Accuracy train: {train_score}")
    print(f"Accuracy test: {test_score}")

    print("Attributes of the learned model:")
    if modelchoice == "gauss":
        print(f"theta: {model.theta_}")
        print(f"sigma: {model.sigma_}")
    if modelchoice == "lda":
        print(f"means: {model.means_}")
    if modelchoice == "log":
        print(f"coef: {model.coef_}")
    if modelchoice == "svc":
        print(f"coef: {model.coef_}")

    return train_score, test_score


def main():
    part1_ = False
    check_against = False
    classifiers = ["gauss", "lda", "log", "svc"]

    # "mi", "gauss"
    data = "mi"

    gauss_samples = []
    # 2, 3, 0.5, 0.5, 0.45
    # 5, 3, 1.0, 0.5, -0.45
    # 3, 2, 0.5, 0.5, 0
    # [meanx, meany, variance_x, variance_y, covariance]
    gauss_samples.append([0, 6, 5, 5, 3])
    gauss_samples.append([6, 0, 5, 5, -3])
    gauss_samples.append([14, 5, 5, 5, 3])

    if data == "mi":
        if check_against:
            print("Cant do check against thing on mi, only on gauss")
            return
        train_features, train_labels = mi_data()
    else:
        train_features, train_labels = gauss_sampling(gauss_samples)
        if check_against:
            test_features, test_labels = gauss_sampling(gauss_samples)

    train_scores = []
    test_scores = []
    for modelchoice in classifiers:
        print("")
        print(f" classifier: {modelchoice}")
        if part1_:
            print(f" data_depr: {data}")
            # Learning a classifier -- uncomment to select the classification model to use.
            if modelchoice == "gauss":
                model = GaussianNB()
            if modelchoice == "lda":
                model = LinearDiscriminantAnalysis()
            if modelchoice == "log":
                model = LogisticRegression()
            if modelchoice == "svc":
                model = SVC(kernel='linear')
            model.fit(train_features, train_labels)

            train_score = part1(model, data, modelchoice, train_features, train_labels, "train")
            if check_against:
                test_score = part1(model, data, modelchoice, test_features, test_labels, "test")
                train_scores.append(train_score)
                test_scores.append(test_score)
        else:
            train_score, test_score = part2(modelchoice)
            train_scores.append(train_score)
            test_scores.append(test_score)

    if not part1_:
        # Which of the input features are most important for the prediction?
        print("")
        print(f" data_depr: Boston")
        print("Scores: gauss, lda, log, svc")
        print(f"train scores: {train_scores}")
        print(f"test scores: {test_scores}")

    if check_against:
        print("")
        print("Scores: gauss, lda, log, svc")
        print(f"train scores: {train_scores}")
        print(f"test scores: {test_scores}")

        for index1, train_score1 in enumerate(train_scores):
            for index2, train_score2 in enumerate(train_scores):
                if train_score1 > train_score2:
                    if test_scores[index2] > test_scores[index1]:
                        print("Found match!")
                        print(f"{classifiers[index1]} was better in training than {classifiers[index2]}")
                        print(f"but {classifiers[index2]} was better in testing than {classifiers[index1]}")
                        print(f"train result for {classifiers[index1]}: {train_score1}")
                        print(f"train result for {classifiers[index2]}: {train_score2}")
                        print(f"test result for {classifiers[index1]}: {test_scores[index1]}")
                        print(f"test result for {classifiers[index2]}: {test_scores[index2]}")
                        print("")


if __name__ == "__main__":
    main()
