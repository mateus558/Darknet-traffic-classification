import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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
    accs = []
    step = 0
    pbar = tqdm(skf.split(X, y), total=n_splits)
    for train_index, test_index in pbar:
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model = model.fit(x_train_fold, y_train_fold)
        y_pred_fold = model.predict(x_test_fold)
        class_report = classification_report(y_test_fold, y_pred_fold, target_names=labels,
                                             output_dict=True)
        acc = accuracy_score(y_test_fold, y_pred_fold)
        accs.append(acc)
        pbar.set_postfix({'Test acc. #{}'.format(step): acc})
        classifier_reports.append(class_report)
        step += 1

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
    final_report["accuracies"] = accs
    
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

def show_confusion_matrix(model, X, y, labels, vmax=None, rotation=(0,0), fname=None, display_labels=None, figsize=(8,8)):
    y_pred = model.predict(X)
    conf = confusion_matrix(y, y_pred)
    n_spaces = max([len(label) for label in labels])
    print("Test accuracy: %0.2f%%\n"%(np.sum((y_pred==y))/y.size*100))
    print("\nConfusion matrix:")
    print(conf)
    print()
    accs = {}
    for i in range(conf.shape[0]):
        accs[labels[i]] = conf[i, i] / np.sum(conf[:, i]) * 100
        print(labels[i],":"," "*(n_spaces- len(labels[i])) ," %0.2f%%"%(accs[labels[i]]))

    if display_labels == None:
        df_conf = pd.DataFrame(conf, index=labels, columns=labels)
    else:
        df_conf = pd.DataFrame(conf, index=display_labels, columns=display_labels)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_conf, cbar=True, annot=True, cmap='Blues', fmt='g', vmax=vmax)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.yticks(rotation=rotation[0])
    plt.xticks(rotation=rotation[1])
    plt.ylabel("Real")
    plt.xlabel("Predição")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return accs

def evaluate_model(model, X_train, y_train, X_test, y_test, labels, n_splits=10, figsize=(10, 10), rotation=(35, 25),
                   fname="conf_mat", display_labels=None, vmax=None):
    kfold_report = kfold_validation(model, X_train, y_train, n_splits=n_splits)
    show_kfold_report(kfold_report, labels)
    accs = show_confusion_matrix(model, X_test, y_test, labels, figsize=figsize, display_labels=display_labels,
                          vmax=vmax, rotation=rotation, fname=fname)
    return kfold_report, accs

def metrics_polar_plot(report, labels, metrics, display, show_legend=True, figsize=None, fname=None, bbox_to_anchor=None):
    mat = np.zeros((len(labels), len(metrics)))
    df_metrics = pd.DataFrame(mat, index=labels,columns=metrics)
    for label in labels:
        for metric in metrics:
            df_metrics.loc[label, metric] = 1-report[label][metric.lower()]

    angles = [(n / float(len(labels)) * 2 * math.pi) for n in range(1,len(labels)+1)]
    angles += angles[:1]

    i = 0
    max_max = 0
    min_min = 1
    fig = plt.figure(figsize=figsize)
    for metric in df_metrics.columns:
        if min(df_metrics[metric]) < min_min:
            min_min = min(df_metrics[metric])
        if max(df_metrics[metric]) > max_max:
            max_max = max(df_metrics[metric])
        plt.polar(angles, df_metrics[metric].tolist()+df_metrics[metric].tolist()[:1], label=display[i])
        i+=1

    plt.xticks(angles[:-1], labels)
    plt.yticks(np.round(np.linspace(min_min, max_max, 5),2))
    if show_legend:
        plt.legend(loc='lower right', bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()

def accuracy_polar_plot(accs, labels, models, display=None, show_legend=True, figsize=None, fname=None,bbox_to_anchor=None):
    mat = np.zeros((len(labels), len(models)))
    df_metrics = pd.DataFrame(mat, index=labels,columns=models)
    for label in labels:
        for model in models:
            df_metrics.loc[label, model] = 100-accs[model][label]
    angles = [(n / float(len(labels)) * 2 * math.pi) for n in range(1,len(labels)+1)]
    angles += angles[:1]

    i = 0
    max_max = 0
    min_min = 1
    fig = plt.figure(figsize=figsize)
    for model in df_metrics.columns:
        if min(df_metrics[model]) < min_min:
            min_min = min(df_metrics[model])
        if max(df_metrics[model]) > max_max:
            max_max = max(df_metrics[model])
        if display is not None:
            plt.polar(angles, df_metrics[model].tolist()+df_metrics[model].tolist()[:1], label=display[i])
        else:
            plt.polar(angles, df_metrics[model].tolist()+df_metrics[model].tolist()[:1], label=models[i])
        i+=1

    plt.xticks(angles[:-1], labels)
    plt.yticks(np.round(np.linspace(min_min, max_max, 5),2))
    if show_legend:
        plt.legend(loc='lower right', bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()