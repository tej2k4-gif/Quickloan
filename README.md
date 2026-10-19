# QuickLoan: The Great Model Bake-Off

| File | What it is |
|---|---|
| `QuickLoan_Ensemble_BakeOff.ipynb` | The full notebook, already run with outputs visible. Steps 1–5: weak single models → Voting, Bagging, Boosting (incl. XGBoost), Stacking → leaderboard (accuracy, recall, money saved) → threshold tuning and robustness check → verdict. Charts are interactive Plotly, with a static image fallback. |
| `app.py` | Upgraded Streamlit dashboard (stretch goal) with a decision card, a money card and a threshold lab. Reads the saved cut-off. |
| `loan_model.joblib` | The deployed champion: the Stacking ensemble, its column order and its 0.14 cut-off. |
| `QuickLoan_Dashboard.html` | Stand-alone interactive leaderboard. Open it in any browser; no install needed. |
| `loans.csv` | The data provided with the brief. |

## Headline result (1,500 held-out loans)
- **Brief's 50 % rule:** Gradient Boosting is champion, saving AED 4.77 M vs approving everyone. Accuracy varies by only 1.5 points across models, while money saved varies by AED 2.8 M.
- **Money-tuned cut-off (chosen on training folds only):** Stacking is champion, saving **AED 6.01 M**. It catches 98 of 188 defaulters with 86.0 % accuracy.

## Run it
```bash
pip install pandas numpy scikit-learn xgboost plotly streamlit joblib
jupyter notebook QuickLoan_Ensemble_BakeOff.ipynb   # re-running takes ~3–4 minutes
streamlit run app.py                                 # needs app.py, loan_model.joblib and loans.csv in one folder
```
`loan_model.joblib` was saved with scikit-learn 1.9 and XGBoost 3.4. If your versions differ and the app won't load the model, run the notebook once to regenerate the file.
