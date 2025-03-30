# AUROC

"""
==================================================
Multiclass Receiver Operating Characteristic (ROC)
==================================================

This example describes the use of the Receiver Operating Characteristic (ROC)
metric to evaluate the quality of multiclass classifiers.

ROC curves typically feature true positive rate (TPR) on the Y axis, and false
positive rate (FPR) on the X axis. This means that the top left corner of the
plot is the "ideal" point - a FPR of zero, and a TPR of one. This is not very
realistic, but it does mean that a larger area under the curve (AUC) is usually
better. The "steepness" of ROC curves is also important, since it is ideal to
maximize the TPR while minimizing the FPR.

ROC curves are typically used in binary classification, where the TPR and FPR
can be defined unambiguously. In the case of multiclass classification, a notion
of TPR or FPR is obtained only after binarizing the output. This can be done in
2 different ways:

- the One-vs-Rest scheme compares each class against all the others (assumed as
  one);
- the One-vs-One scheme compares every unique pairwise combination of classes.

In this example we explore both schemes and demo the concepts of micro and macro
averaging as different ways of summarizing the information of the multiclass ROC
curves.

.. note::

    See :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py` for
    an extension of the present example estimating the variance of the ROC
    curves and their respective AUC.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Load and prepare data
# =====================
#
# We import the :ref:`iris_dataset` which contains 3 classes, each one
# corresponding to a type of iris plant. One class is linearly separable from
# the other 2; the latter are **not** linearly separable from each other.
#
# Here we binarize the output and add noisy features to make the problem harder.

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
target_names = iris.target_names
X, y = iris.data, iris.target
y = iris.target_names[y]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
n_classes = len(np.unique(y))
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

# %%
# We train a :class:`~sklearn.linear_model.LogisticRegression` model which can
# naturally handle multiclass problems, thanks to the use of the multinomial
# formulation.

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# %%
# One-vs-Rest multiclass ROC
# ==========================
#
# The One-vs-the-Rest (OvR) multiclass strategy, also known as one-vs-all,
# consists in computing a ROC curve per each of the `n_classes`. In each step, a
# given class is regarded as the positive class and the remaining classes are
# regarded as the negative class as a bulk.
#
# .. note:: One should not confuse the OvR strategy used for the **evaluation**
#     of multiclass classifiers with the OvR strategy used to **train** a
#     multiclass classifier by fitting a set of binary classifiers (for instance
#     via the :class:`~sklearn.multiclass.OneVsRestClassifier` meta-estimator).
#     The OvR ROC evaluation can be used to scrutinize any kind of classification
#     models irrespectively of how they were trained (see :ref:`multiclass`).
#
# In this section we use a :class:`~sklearn.preprocessing.LabelBinarizer` to
# binarize the target by one-hot-encoding in a OvR fashion. This means that the
# target of shape (`n_samples`,) is mapped to a target of shape (`n_samples`,
# `n_classes`).

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

# %%
# We can as well easily check the encoding of a specific class:

label_binarizer.transform(["virginica"])

# %%
# ROC curve showing a specific class
# ----------------------------------
#
# In the following plot we show the resulting ROC curve when regarding the iris
# flowers as either "virginica" (`class_id=2`) or "non-virginica" (the rest).

class_of_interest = "virginica"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

# %%
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

display = RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)",
)

# %%
# ROC curve using micro-averaged OvR
# ----------------------------------
#
# Micro-averaging aggregates the contributions from all the classes (using
# :func:`numpy.ravel`) to compute the average metrics as follows:
#
# :math:`TPR=\frac{\sum_{c}TP_c}{\sum_{c}(TP_c + FN_c)}` ;
#
# :math:`FPR=\frac{\sum_{c}FP_c}{\sum_{c}(FP_c + TN_c)}` .
#
# We can briefly demo the effect of :func:`numpy.ravel`:

print(f"y_score:\n{y_score[0:2,:]}")
print()
print(f"y_score.ravel():\n{y_score[0:2,:].ravel()}")

# %%
# In a multi-class classification setup with highly imbalanced classes,
# micro-averaging is preferable over macro-averaging. In such cases, one can
# alternatively use a weighted macro-averaging, not demonstrated here.

display = RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
)

# %%
# In the case where the main interest is not the plot but the ROC-AUC score
# itself, we can reproduce the value shown in the plot using
# :class:`~sklearn.metrics.roc_auc_score`.

from sklearn.metrics import roc_auc_score

micro_roc_auc_ovr = roc_auc_score(
    y_test,
    y_score,
    multi_class="ovr",
    average="micro",
)

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

# %%
# This is equivalent to computing the ROC curve with
# :class:`~sklearn.metrics.roc_curve` and then the area under the curve with
# :class:`~sklearn.metrics.auc` for the raveled true and predicted classes.

from sklearn.metrics import auc, roc_curve

# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
