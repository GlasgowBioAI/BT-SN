"""Shared survival evaluation with within-fold averaging across model seeds."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold


def cross_validated_cindex(run_embeddings, survival, n_splits=5, random_state=42):
    """Average seed-specific C-indices within each fold, then summarize folds."""
    common = survival.index
    for embedding in run_embeddings:
        common = common.intersection(embedding.index)
    survival = survival.loc[common, ["time", "status"]]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_values = []
    for train_index, test_index in folds.split(survival):
        seed_values = []
        for embedding in run_embeddings:
            frame = embedding.loc[common].copy()
            frame[["time", "status"]] = survival
            model = CoxPHFitter()
            model.fit(frame.iloc[train_index], "time", "status", show_progress=False)
            risk = model.predict_partial_hazard(frame.iloc[test_index].drop(columns=["time", "status"]))
            test = frame.iloc[test_index]
            seed_values.append(concordance_index(test["time"], -risk, test["status"]))
        fold_values.append(float(np.mean(seed_values)))
    return np.asarray(fold_values)


def heldout_risk_groups(train_runs, test_runs, train_survival, test_survival):
    """Fit one Cox model per seed and average held-out risk before median splitting."""
    train_common, test_common = train_survival.index, test_survival.index
    for frame in train_runs: train_common = train_common.intersection(frame.index)
    for frame in test_runs: test_common = test_common.intersection(frame.index)
    risks = []
    for train_embedding, test_embedding in zip(train_runs, test_runs):
        train = train_embedding.loc[train_common].copy()
        train[["time", "status"]] = train_survival.loc[train_common, ["time", "status"]]
        model = CoxPHFitter(); model.fit(train, "time", "status", show_progress=False)
        risks.append(model.predict_partial_hazard(test_embedding.loc[test_common]).to_numpy())
    mean_risk = pd.Series(np.mean(risks, axis=0), index=test_common, name="risk")
    groups = mean_risk >= mean_risk.median()
    outcomes = test_survival.loc[test_common, ["time", "status"]]
    high, low = outcomes.loc[groups], outcomes.loc[~groups]
    p_value = logrank_test(high["time"], low["time"], high["status"], low["status"]).p_value
    return mean_risk, groups, float(p_value)
