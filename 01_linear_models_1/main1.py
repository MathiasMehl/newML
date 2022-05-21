from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC


# A small utility we shall need later:
def maxpos(A):
    '''
    Takes an n x k array A, and returns 1-dim n array where the i'th
    entry is the index of column in A where the i'th row of A has its
    maximal value (application: turns a probabilitiy distribution over
    k classes for n instances into a single prediction)
    '''
    return np.argmax(A, axis=1)


def xx_yy(irisdata):
    maxval1 = np.max(irisdata.data[:, feat1])
    maxval2 = np.max(irisdata.data[:, feat2])

    xx, yy = np.meshgrid(np.arange(0, maxval1 + 1, 0.02), np.arange(0, maxval2 + 1, 0.02))
    return xx, yy


def plot_knn_curve(features_train, features_test,
                   labels_train, labels_test, features):
    max = 40
    x = range(1, max)
    train_scores = []
    test_scores = []
    if len(features) == 4:
        features = "All features"

    for n in x:
        model = KNeighborsClassifier(n).fit(features_train, labels_train)

        pred_labels_train = model.predict(features_train)
        pred_labels_test = model.predict(features_test)

        train_scores.append(accuracy_score(labels_train, pred_labels_train))
        test_scores.append(accuracy_score(labels_test, pred_labels_test))

    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.axis([1, 40, 0.65, 1])
    plt.title(f"KNN accuracy train data: {features}")
    plt.plot(range(1, max), train_scores)
    plt.show()

    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.axis([1, 40, 0.65, 1])
    plt.title(f"KNN accuracy test data: {features}")
    plt.plot(range(1, max), test_scores)
    plt.show()


def display_results(meshclasses, xx, yy, irisdata, model_choice, neighbours):
    meshclasses = meshclasses.reshape(xx.shape)
    plt.contourf(xx, yy, meshclasses, [-0.1, 0.1, 1, 2], colors=('tomato', 'lightblue', 'lightgreen'))
    plt.scatter(irisdata.data[:, feat1], irisdata.data[:, feat2], c=irisdata.target,
                cmap=mpl.colors.ListedColormap(['r', 'b', 'g']))
    plt.xlabel(irisdata['feature_names'][feat1])
    plt.ylabel(irisdata['feature_names'][feat2])
    if model_choice == "knn":
        title = model_choice + str(neighbours)
    else:
        title = model_choice
    plt.title(title)
    plt.show()


def fitted_model(irisdata, feat1, feat2, model_choice, neighbours, split, features_train, features_test,
                 labels_train, labels_test, knn_curve, features):
    if model_choice == "knn":
        if knn_curve:
            feature_names = []
            for feature in features:
                feature_names.append(irisdata['feature_names'][feature])
            plot_knn_curve(features_train, features_test,
                           labels_train, labels_test, feature_names)
            return
        else:
            print(f"Neighbours: {neighbours}")
            model = KNeighborsClassifier(n_neighbors=neighbours)
    elif model_choice == "log":
        model = LogisticRegression()
    elif model_choice == "svc":
        model = SVC(kernel='linear')
    else:
        return

    if split:
        model.fit(features_train, labels_train)
    else:
        model.fit(irisdata.data[:, features], irisdata.target)
    return model


def predict(model, features_train, features_test, labels_train, labels_test, confusion_matrix):
    pred_labels_train = model.predict(features_train)
    pred_labels_test = model.predict(features_test)

    print("Accuracy train: {}".format(accuracy_score(labels_train, pred_labels_train)))
    print("Accuracy test: {}".format(accuracy_score(labels_test, pred_labels_test)))

    if confusion_matrix:
        metrics.plot_confusion_matrix(model, features_test, labels_test)
        plt.show()


def predict_linear(model, features_train, features_test, labels_train, labels_test, confusion_matrix):
    train_z = model.predict(features_train)
    test_z = model.predict(features_test)

    pred_labels_train = maxpos(train_z)
    pred_labels_test = maxpos(test_z)

    print("Accuracy train: {}".format(accuracy_score(labels_train, pred_labels_train)))
    print("Accuracy test: {}".format(accuracy_score(labels_test, pred_labels_test)))

    if confusion_matrix:
        metrics.plot_confusion_matrix(model, features_test, labels_test)
        plt.show()


def getonehotclasses_no_split(irisdata):
    onehotclasses = np.zeros((irisdata.target.size, 3))
    for i in range(irisdata.target.size):
        onehotclasses[i][irisdata.target[i]] = 1.0
    return onehotclasses


def getonehotclasses_split(data):
    onehotclasses = np.zeros((data.size, 3))
    for i in range(data.size):
        onehotclasses[i][data[i]] = 1.0
    return onehotclasses


def linear(irisdata, xx, yy, features_train, features_test, labels_train, labels_test, all_data, confusion_matrix):
    model = LinearRegression()
    if split:
        onehotclasses = getonehotclasses_split(labels_train)
        model.fit(features_train, onehotclasses)
        predict_linear(model, features_train, features_test, labels_train, labels_test, confusion_matrix)
        return
    else:
        onehotclasses = getonehotclasses_no_split(irisdata)
        model.fit(irisdata.data[:, features], onehotclasses)
        if all_data:
            Z = model.predict(irisdata.data)
            pred_labels = maxpos(Z)
            print("Accuracy: {}".format(accuracy_score(irisdata.target, pred_labels)))
            if confusion_matrix:
                metrics.plot_confusion_matrix(model, irisdata.data, irisdata.target)
                plt.show()
            return
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            return maxpos(Z)


def main(feat1, feat2, print_data, model_choice, neighbours, split, features, knn_curve, confusion_matrix):
    irisdata = load_iris()
    xx, yy = xx_yy(irisdata)

    feature_names = []
    for feature in features:
        feature_names.append(irisdata['feature_names'][feature])
    print(f"Features: {feature_names}")

    if print_data:
        plt.scatter(irisdata.data[:, feat1], irisdata.data[:, feat2], c=irisdata.target)
        plt.xlabel(irisdata['feature_names'][feat1])
        plt.ylabel(irisdata['feature_names'][feat2])
        plt.show()

    features_train, features_test, labels_train, labels_test = train_test_split(irisdata.data[:, features],
                                                                                irisdata.target,
                                                                                test_size=0.30, train_size=0.70)

    if model_choice == "linear":
        meshclasses = linear(irisdata, xx, yy, features_train, features_test, labels_train, labels_test, all_data,
                             confusion_matrix)
        if all_data:
            return
    else:
        model = fitted_model(irisdata, feat1, feat2, model_choice, neighbours, split, features_train,
                             features_test, labels_train, labels_test, knn_curve, features)
        if knn_curve:
            return
        if split:
            predict(model, features_train, features_test, labels_train, labels_test, confusion_matrix)
        else:
            if all_data:
                pred_labels = model.predict(irisdata.data[:, features])
                print("Accuracy: {}".format(accuracy_score(irisdata.target, pred_labels)))

                if confusion_matrix:
                    metrics.plot_confusion_matrix(model, irisdata.data, irisdata.target)
                    plt.show()

                return
            else:
                meshclasses = model.predict(np.c_[xx.ravel(), yy.ravel()])

    if not split:
        display_results(meshclasses, xx, yy, irisdata, model_choice, neighbours)


if __name__ == "__main__":
    print_data = True
    confusion_matrix = False

    # "knn", "linear", "log", "svc"
    model_choice = "linear"
    neighbours = 1
    knn_curve = False

    split = False
    all_data = True

    pairs = [[2,3]]

    for pair in pairs:
        print("\n")
        feat1 = pair[0]
        feat2 = pair[1]
        features = [feat1, feat2]

        if all_data:
            features = [0, 1, 2, 3]

        print(f"Splitting data: {split}")
        # print(f"plotting knn_curve: {knn_curve}")
        if split:
            print(f"Learning model: {model_choice}\n"
                  f"Features: {features}")
        else:
            print(f"Learning model: {model_choice}\n "
                  f"Features: {features}")

        main(feat1, feat2, print_data, model_choice, neighbours, split, features, knn_curve, confusion_matrix)
        if all_data:
            break
