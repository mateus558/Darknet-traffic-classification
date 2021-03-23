from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from .classification import *

def select_features_rfecv(model, X, y, min_feats=1,step=2, random_state=None):
    rfe = RFECV(model, min_features_to_select=min_feats, step=step, cv=StratifiedKFold(10, shuffle=True, random_state=random_state), scoring='accuracy', n_jobs=4, verbose=3)
    rfe.fit(X, y)
    return rfe

def final_evaluation_rfe(model, X, y, labels, rfe, random_state=42):
    X_selected = rfe.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=random_state)
    kfold_report = kfold_validation(model, X_train, y_train, n_splits=10)
    show_kfold_report(kfold_report, np.unique(y_train))
    show_confusion_matrix(model, X_test, y_test, labels)


def summarize_feats(rfe, model, columns, to_remove=[]):
    columns = [column for column in columns if column not in to_remove]
    columns = [columns[i] for i in range(len(columns)) if (i < len(rfe.support_))
               and rfe.support_[i]]
    features = zip(columns, model.feature_importances_)
    feat_import = [(feat, importance) for feat, importance in features]
    feat_import.sort(key=lambda x: x[1], reverse=True)

    n_spaces = max([len(feat) for feat, _ in feat_import])
    head = "Feature" + str(" " * (n_spaces - len("Feature")) + "\tImportance")
    print(head)
    print("-" * (len(head) + 5))
    for feature, importance in feat_import:
        print(f"%s:" % (feature), " " * (n_spaces - len(feature)), "\t%0.4f" % (importance))

    return columns
