import os
import pickle
import uuid
import warnings

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from keel import find_datasets, load_dataset
from eus import EUS, MEUS, metrics


RESULTS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'results')
OBJECTS_DIR = os.path.join(RESULTS_DIR, 'objects')

if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

if not os.path.isdir(OBJECTS_DIR):
    os.mkdir(OBJECTS_DIR)

CLASSIFIERS = {
    ('CART', DecisionTreeClassifier()),
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC()),
}

def get_estimators(classifiers=CLASSIFIERS):
    for clf_name, clf in classifiers:
        estimators = (
            ("None", clf_name, clf),
            ("EUS-GMS", clf_name, EUS(clf, metrics.soo_score_gmean1)),
            ("EUS-GM", clf_name, EUS(clf, metrics.soo_score_gmean2)),
            ("MEUS-SS", clf_name, MEUS(clf, metrics.moo_score_sns_spc, 2)),
            ("MEUS-SP", clf_name, MEUS(clf, metrics.moo_score_sns_ppv, 2)),
            ("MEUS-SSP", clf_name, MEUS(clf, metrics.moo_score_sns_spc_ppv, 3)),
            ("NM", clf_name, Pipeline([('smt', NearMiss()), ('clf', clf)])),
            ("RUS", clf_name, Pipeline([('smt', RandomUnderSampler()), ('clf', clf)])),
        )

        for estimator in estimators:
            yield estimator

RANDOM_STATE = 90210
FOLDING = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=RANDOM_STATE)


def store_estimator(est):
    while (clf_uuid := f"{str(uuid.uuid4())}.pkl") in os.listdir(OBJECTS_DIR):
        warnings.warn(f"{clf_uuid} already exist - generating new")

    with open(os.path.join(OBJECTS_DIR, clf_uuid), 'wb') as fp:
        pickle.dump(est, fp)

    return clf_uuid

def main():
    for data_name in find_datasets():

        if os.path.exists(os.path.join(RESULTS_DIR, f"{data_name}.pkl")):
            print(f"{data_name} already tested")
            continue

        X, y = load_dataset(data_name, return_X_y=True)
        X = StandardScaler().fit_transform(X, y)

        dataset_resuts = []

        for idx, (train_index, test_index) in enumerate(FOLDING.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for prc_name, est_name, estimator in get_estimators():
                print(f"[{data_name}][{idx}] - {prc_name}+{est_name}")

                clf = clone(estimator)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

                # WARNING: Stroing estimators can use much of your disk space.
                # clf_uuid = store_estimator(clf)

                dataset_resuts.append({
                    "Fold": idx,
                    "Processing": prc_name,
                    "Classifier": est_name,
                    'cm': cm,
                    # "clf_obj": clf_uuid,
                })

        with open(os.path.join(RESULTS_DIR, f"{data_name}.pkl"), 'wb') as fp:
            pickle.dump(dataset_resuts, fp)


if __name__ == '__main__':
    main()
