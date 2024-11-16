import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

with open('/storage/ghadia/DeepRC2/results/T1D_AC_clip/puneet_folds_outputs_c1.pkl', 'rb') as f:
    cohort1_results = pkl.load(f)
    cohort1_results = {i: list(cohort1_results[i]) for i in range(5)}

for i in range(5):
    cohort1_results[i][1] = cohort1_results[i][1].cpu().numpy()

with open('/storage/ghadia/DeepRC2/results/T1D_AC_clip/puneet_folds_outputs_c23.pkl', 'rb') as f:
    cohorts2_3_results = pkl.load(f)
    cohorts2_3_results = {i: list(cohorts2_3_results[i]) for i in range(5)}

for i in range(5):
    cohorts2_3_results[i][1] = cohorts2_3_results[i][1].cpu().numpy()

# redistribute the validation and training splits to not make any model more advantageous than another
with open('/storage/ghadia/DeepRC2/deeprc/datasets/T1D/used_t1d_ids.pkl', 'rb') as sfh:
    split_inds = pkl.load(sfh)

train_inds = np.concatenate([subset[:207] for subset in split_inds])
val_inds = np.concatenate([subset[207:] for subset in split_inds])

targets1 = cohort1_results[0][2].cpu().numpy()
sample_ids1 = cohort1_results[0][0]
results = {}

# getting the models and the predictions

raw_data = np.hstack([cohort1_results[i][1] for i in range(5)])
training_data = raw_data[train_inds]
val_data = raw_data[val_inds]

### get models
results['avg_raw_ensemble'] = val_data.mean(1)
results['avg_prediction_ensemble'] = (val_data > 0).mean(1)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ['l2', 'l1'],
    'solver': ['liblinear']  # 'liblinear' is a good choice for small datasets and l1/l2 penalties
}

validation_fold = np.zeros(1298)
validation_fold[val_inds] = 0  # last 20% for validation
validation_fold[train_inds] = -1  # first 80% for training

ps = PredefinedSplit(test_fold=validation_fold)

raw_logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
grid_search = GridSearchCV(estimator=raw_logreg, param_grid=param_grid, cv=ps, scoring='roc_auc')
grid_search.fit(raw_data, targets1)
raw_logreg = grid_search.best_estimator_
results['raw_logreg'] = raw_logreg.predict_proba(val_data)[:, 1]

thresholded_logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
grid_search = GridSearchCV(estimator=thresholded_logreg, param_grid=param_grid, cv=ps, scoring='roc_auc')
grid_search.fit(raw_data > 0, targets1)
thresholded_logreg = grid_search.best_estimator_
results['thresholded_logreg'] = thresholded_logreg.predict_proba(val_data > 0)[:, 1]
### end get models

# evaluate the models

metrics = {}
for name, predictions in results.items():
    metrics[name] = roc_auc_score(targets1[val_inds], predictions)

best_model = max(metrics, key=metrics.get)  # turned out to be raw_logreg

# save a dict with sample ids, predictions and targets
all_results = raw_logreg.predict_proba(raw_data)[:, 1]
all_results = pd.DataFrame(all_results)
all_results['sample_id'] = sample_ids1
all_results['target'] = targets1
all_results.to_csv(f'/storage/ghadia/DeepRC2/results/T1D_AC_clip/ensemble_results_c1.csv', index=False)

print()
# apply best model to cohorts 2+3

if True:
    sample_ids23 = cohorts2_3_results[0][0]

    test_data = np.hstack([cohorts2_3_results[i][1] for i in range(5)])
    # test_predictions = test_data.mean(1)
    test_predictions = raw_logreg.predict_proba(test_data)[:, 1]

    test_targets = cohorts2_3_results[0][2].cpu().numpy()
    test_metrics = {"roc_auc": roc_auc_score(test_targets, test_predictions),
                    "roc_pr": average_precision_score(test_targets, test_predictions)}

    print(f'test AUC: {test_metrics["roc_auc"]:.3f}, test PR AUC: {test_metrics["roc_pr"]:.3f}')

    # save a dict with sample ids, predictions and targets
    all_results = pd.DataFrame({"sample_id": sample_ids23, "target": test_targets.squeeze(), "prediction": test_predictions})
    all_results.to_csv(f'/storage/ghadia/DeepRC2/results/T1D_AC_clip/ensemble_results_c23.csv', index=False)
