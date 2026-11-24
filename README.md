# Home Credit: probability of default

A logistic-regression-with-splines credit model on the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) data: hypotheses → EDA → leakage and redundancy audit → feature engineering from the bureau and Home Credit history tables → linearity-driven pipeline → calibration and out-of-fold evaluation → XGBoost challenger.

Everything is in `home_credit_default.ipynb`.


## Results (5-fold, out of fold)

| | Logistic regression | XGBoost challenger | Predict the average |
|---|---|---|---|
| Log-loss | 0.2436 | 0.2383 | 0.2805 |
| ROC AUC | 0.767 (± 0.004) | 0.783 (± 0.004) | 0.5 |
| Average precision | 0.252 | 0.276 | 0.081 |
| Brier score | 0.0673 (± 0.0003) | 0.0662 (± 0.0003) | 0.0742 |

Scores are pooled over all out-of-fold predictions; ± is the standard deviation across the 5 folds. XGBoost's AUC lead (0.016) is about four times that spread.

The logistic regression is calibrated to within 0.3 points in 8 of 10 deciles of predicted risk (worst: 14.7% predicted vs 15.5% observed), with predictions ranging from 1.1% to 28.4%. XGBoost, on the same features and folds, is equally well calibrated; its edge comes from interactions, not from different features.

XGBoost trains on an NVIDIA GPU (`device="cuda"`). Without one, set `"device": "cpu"` in `xgb_params`; it still runs, just slower.
