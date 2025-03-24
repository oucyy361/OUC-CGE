import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc


GT = ["Low", "Mid", "High"]


def read_excel(source, pre_title, pre_proba_title, label_title, conf_titles):
    data = pd.read_excel(source, sheet_name='Sheet1')

    pre = data[pre_title].values
    pre_proba_title = data[pre_proba_title].values
    label = data[label_title].values

    confs = []

    for i in range(len(GT)):
         confs.append(data[f"{conf_titles}{i}"].values)

    return pre, pre_proba_title, label, np.stack(confs, axis=-1)

def read_csv(source, pre_title, pre_proba_title, label_title, conf_titles):
    data = pd.read_csv(source)

    pre = data[pre_title].values
    pre_proba_title = data[pre_proba_title].values
    label = data[label_title].values

    confs = []

    for i in range(len(GT)):
        confs.append(data[f"{conf_titles}i"].values)

    return pre, pre_proba_title, label, confs

def _roc_curve(scores_list, true_list, save_path, names, n_classes=2):

    auc_list = list()
    _true = np.array([[i for i in range(n_classes)]])
    _true = np.repeat(_true, len(scores_list[0]), axis=0)

    plt.figure(figsize=(8, 6))

    for na, y_true, y_scores in zip(names, true_list, scores_list):

        y_true = np.longlong(np.repeat(y_true[:,None], n_classes, axis=-1) == _true)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        all_fpr = np.unique(np.concatenate([roc_curve(y_true[:, i], y_scores[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr, linestyle='--',
                 label=f'{na}: AUC = {macro_auc:.3f}', lw=2)

        auc_list.append(macro_auc)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Macro-ROC Curve (OvR)')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)

    return auc_list

def cal_metrics(root, pre_title, pre_proba_title, label_title, conf_titles):
    results_list = list(i for i in os.listdir(root) if i.endswith('.xlsx'))

    f1_list = []
    acc_list = []
    recall_list = []
    precision_list = []
    results_name = []

    confs_list = []
    labels_list = []

    for r in results_list:
        result_path = os.path.join(root, r)

        name_ = r.split('.')[0]
        results_name.append(name_)

        pre, proba, label, confs = read_excel(result_path, pre_title, pre_proba_title, label_title, conf_titles)

        confs_list.append(confs)
        labels_list.append(label)

        acc_ = accuracy_score(label, pre)
        f1_ = f1_score(label, pre, average='macro')
        recall_ = recall_score(label, pre, average='macro')
        precision_ = precision_score(label, pre, average='macro')

        cm = confusion_matrix(label, pre)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        ConfusionMatrixDisplay(cm, display_labels=GT).plot(cmap='Blues', text_kw={'fontsize':10}, values_format='.3f')
        plt.savefig(os.path.join(root, name_ + '_confusion_matrix.jpeg'), dpi=300)


        acc_list.append(acc_)
        f1_list.append(f1_)
        recall_list.append(recall_)
        precision_list.append(precision_)

    macro_auc_list = _roc_curve(confs_list, labels_list, os.path.join(root, 'macro_roc.jpeg'), results_name, len(GT))

    return acc_list, f1_list, recall_list, precision_list, results_name, macro_auc_list





if __name__ == "__main__":

    ROOT = r"./results path"
    returns_ = cal_metrics(ROOT, "pre_class", "pre_conf", "labels", "preds_vector_")

    print('\n')
    for acc, f1, recall, precision, name, auc in zip(*returns_):
        print("Name:", name, "ACC:", acc, "F1:", f1, "Recall:", recall, "Precision:",precision, "Macro-AUC:", auc)

