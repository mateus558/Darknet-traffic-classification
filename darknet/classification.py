import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

def split_train_target(samples, labels_col):
    X = samples.copy()
    del X[labels_col]
    X = X.values
    y = samples[labels_col].values
    labels = np.unique(samples[labels_col])
    return X, y, labels


def kfold_validation(model, X, y, labels=None, n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifier_reports = []
    for train_index, test_index in tqdm(skf.split(X, y), total=n_splits):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model = model.fit(x_train_fold, y_train_fold)
        y_pred_fold = model.predict(x_test_fold)
        class_report = classification_report(y_test_fold, y_pred_fold, target_names=labels,
                                             output_dict=True)
        classifier_reports.append(class_report)

    final_report = {}
    for report in classifier_reports:
        for label in report.keys():
            if type(report[label]) == dict:
                if label not in final_report:
                    final_report[label] = {}
                for key in report[label].keys():
                    if key not in final_report[label]:
                        final_report[label][key] = 0.0
                    final_report[label][key] += report[label][key]
            else:
                if label not in final_report:
                    final_report[label] = 0.0
                final_report[label] += report[label]

    for label in final_report.keys():
        if type(final_report[label]) == dict:
            for metric in final_report[label].keys():
                final_report[label][metric] /= n_splits
        else:
            final_report[label] /= n_splits
    final_report["n_splits"] = n_splits
    return final_report

def show_kfold_report(report, labels):
    metrics_printed = False
    n_spaces = max([len(label) for label in labels])
    for label in labels:
        if not metrics_printed:
            print(" "*n_spaces, end='')
            for metric in report[label].keys():
                print(f"{metric}", " "*6,end='')
            metrics_printed = True
            print()
        print(f"{label}: ", " "*(n_spaces-len(label)),end='')
        for metric in report[label].keys():
            print("%0.2f\t\t"%(report[label][metric]*100),end='')
        print()
    print("\n\n", "%d-fold Accuracy: %0.2f%%"%(report["n_splits"],report['accuracy']*100))

def show_confusion_matrix(model, X, y, labels):
    y_pred = model.predict(X)
    conf = confusion_matrix(y, y_pred)
    n_spaces = max([len(label) for label in labels])
    print("Test accuracy: %0.2f%%\n"%(np.sum((y_pred==y))/y.size*100))
    print("\nConfusion matrix:")
    print(conf)
    print()
    for i in range(conf.shape[0]):
        print(labels[i],":"," "*(n_spaces- len(labels[i])) ," %0.2f%%"%(conf[i,i]/np.sum(conf[:,i])*100))


def evaluate_model(model, X_train, y_train, X_test, y_test, labels, n_splits=10, figsize=(10, 10), rotation=(35, 25),
                   fname="conf_mat", display_labels=None):
    kfold_report = kfold_validation(model, X_train, y_train, n_splits=n_splits)
    show_kfold_report(kfold_report, labels)
    show_confusion_matrix(model, X_test, y_test, labels)

    fig, ax = plt.subplots(figsize=figsize)
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, ax=ax, colorbar=True,
                          display_labels=display_labels)
    plt.yticks(rotation=rotation[0], va='top')
    plt.xticks(rotation=rotation[1])
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return kfold_report
