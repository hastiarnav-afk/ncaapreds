# %% [markdown]
# # Predicting NCAA Men's Basketball Wins & March Madness Success
# Foundational Mathematics of AI — Course Project
#
# This script is organized into cells (marked with `# %%`) that mirror the
# structure of the final presentation. Open it in VS Code / Spyder / Jupyter
# (via jupytext) to run cell-by-cell, or just run it top to bottom with
# `python cbb_full_analysis.py`.
#
# Sections:
#   1. Setup & Data Load
#   2. Feature Engineering
#   3. Descriptive Statistics & EDA (figures saved to ./figures)
#   4. Train/Test Split (temporal: train 2013-2018, test 2019)
#   5. Model 1 -- Multiple Linear Regression (Wins ~ Four Factors)
#   6. Model 2 -- Logistic Regression (Made NCAA Tournament?)
#   7. Extension -- Random Forest (Sweet 16+ among tournament teams)
#   8. Forecast -- Putting it to the test on the 2019 season
#   9. Results summary export

# %% Section 0: Imports & setup -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
)

import os
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})
NAVY, ORANGE, GRAY, RED, GREEN = "#1f3b57", "#e8743b", "#9aa5b1", "#c0392b", "#2e8b57"

DATA_PATH = "cbb.csv"   # <-- put cbb.csv in the same folder as this script,
                         #     or change this to the full path on your computer

# %% Section 1: Load & clean data ---------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. Place cbb.csv in the same folder as this "
        f"script (or edit DATA_PATH above to point to wherever you saved it)."
    )
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Years covered:", df.YEAR.min(), "-", df.YEAR.max())
print("Missing values (should only be POSTSEASON/SEED for non-tourney teams):")
print(df.isna().sum()[df.isna().sum() > 0])

# %% Section 2: Feature engineering -------------------------------------------
df["MADE_TOURNEY"] = df["POSTSEASON"].notna().astype(int)
df["SWEET16_PLUS"] = (df["Tour_Wins"] >= 2).astype(int)

POSTSEASON_ORDER = ["Missed", "R68", "R64", "R32", "S16", "E8", "F4", "2ND", "Champions"]
df["POSTSEASON_FILLED"] = pd.Categorical(
    df["POSTSEASON"].fillna("Missed"), categories=POSTSEASON_ORDER, ordered=True
)

FOUR_FACTORS = ["EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "ADJ_T"]
# NOTE: BARTHAG is intentionally excluded from every model's feature set below.
# BARTHAG is a log5-type transformation computed directly from ADJOE and ADJDE,
# so including it alongside ADJOE/ADJDE creates severe multicollinearity
# (VIF ~36 for BARTHAG, ~14 and ~12 for ADJOE/ADJDE) with no gain in accuracy.
# We still use BARTHAG as a descriptive/EDA variable (e.g., the boxplot-by-stage
# chart) since it is an excellent single-number summary -- just not as a
# regression input alongside the two metrics it is built from.
POWER_FEATS  = ["ADJOE", "ADJDE", "ADJ_T"]
EXT_FEATS    = ["ADJOE", "ADJDE", "ADJ_T", "WAB"]

print(f"\nTournament qualification rate: {df.MADE_TOURNEY.mean():.1%}")
print(f"Sweet16+ rate among tourney teams: {df.loc[df.MADE_TOURNEY==1,'SWEET16_PLUS'].mean():.1%}")

# %% Section 3: Descriptive Statistics & EDA ==================================

# --- 3.1 Summary table of core variables (with units noted in the guide) ----
summary_cols = ["W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D", "TOR", "TORD",
                 "ORB", "DRB", "FTR", "FTRD", "ADJ_T", "WAB"]
desc = df[summary_cols].describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
desc.to_csv(f"{FIG_DIR}/../descriptive_stats_table.csv")
print(desc.round(2))

# --- 3.2 Distribution of season wins -----------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["W"], bins=np.arange(0, 42, 2), color=NAVY, edgecolor="white", alpha=0.9)
ax.axvline(df["W"].mean(), color=ORANGE, linestyle="--", linewidth=2,
           label=f"Mean = {df['W'].mean():.1f} wins")
ax.set_title("Distribution of Season Wins\n(2,455 team-seasons, 2013-2019)")
ax.set_xlabel("Wins (W) — includes conference & NCAA tournament games")
ax.set_ylabel("Number of teams")
ax.legend(frameon=False)
fig.savefig(f"{FIG_DIR}/01_wins_distribution.png")
plt.close(fig)

# --- 3.3 The efficiency "quadrant" scatter -----------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
missed = df[df.MADE_TOURNEY == 0]
made   = df[df.MADE_TOURNEY == 1]
ax.scatter(missed.ADJOE, missed.ADJDE, s=18, color=GRAY, alpha=0.5, label="Did not make tournament")
sc = ax.scatter(made.ADJOE, made.ADJDE, s=32, c=made.Tour_Wins, cmap="YlOrRd",
                 edgecolor="black", linewidth=0.3, label="Made tournament", vmin=0, vmax=6)
ax.invert_yaxis()  # lower ADJDE (fewer points allowed) = better defense, so put "better" at top
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Tournament wins")
ax.set_xlabel("Adjusted Offensive Efficiency (ADJOE)\npoints scored per 100 possessions, opponent-adjusted")
ax.set_ylabel("Adjusted Defensive Efficiency (ADJDE)\npoints allowed per 100 possessions  (lower = better defense, axis flipped)")
ax.set_title("Elite Teams Live in the Top-Right:\nGreat Offense AND Great Defense")
ax.legend(loc="lower right", frameon=True, fontsize=10)
fig.savefig(f"{FIG_DIR}/02_efficiency_quadrant.png")
plt.close(fig)

# --- 3.4 Correlation heatmap --------------------------------------------------
corr_vars = FOUR_FACTORS + ["ADJOE", "ADJDE", "BARTHAG", "WAB", "W"]
corr = df[corr_vars].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_vars))); ax.set_xticklabels(corr_vars, rotation=45, ha="right")
ax.set_yticks(range(len(corr_vars))); ax.set_yticklabels(corr_vars)
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(v) > 0.6 else "black")
ax.set_title("Correlation Matrix — Four Factors, Efficiency Metrics, and Wins")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
fig.savefig(f"{FIG_DIR}/03_correlation_heatmap.png")
plt.close(fig)

# --- 3.5 BARTHAG by postseason stage (validation of the power rating) -------
fig, ax = plt.subplots(figsize=(10, 6))
groups = [df.loc[df.POSTSEASON_FILLED == cat, "BARTHAG"].dropna() for cat in POSTSEASON_ORDER]
bp = ax.boxplot(groups, tick_labels=POSTSEASON_ORDER, patch_artist=True, showfliers=False)
for patch, cat in zip(bp["boxes"], POSTSEASON_ORDER):
    patch.set_facecolor(NAVY if cat != "Missed" else GRAY)
    patch.set_alpha(0.75)
ax.set_title("Power Rating (BARTHAG) Rises Monotonically\nwith How Far a Team Advances")
ax.set_xlabel("Final postseason result")
ax.set_ylabel("BARTHAG (est. win probability vs. average D-I team)")
fig.savefig(f"{FIG_DIR}/04_barthag_by_stage.png")
plt.close(fig)

# --- 3.6 PCA visualization ----------------------------------------------------
pca_vars = FOUR_FACTORS + ["ADJOE", "ADJDE"]
X_pca_raw = df[pca_vars].values
X_pca_std = StandardScaler().fit_transform(X_pca_raw)

# Manual PCA via eigendecomposition of the covariance matrix (foundational-math view)
cov = np.cov(X_pca_std, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
explained_var_manual = eigvals / eigvals.sum()
scores_manual = X_pca_std @ eigvecs[:, :2]

# sklearn cross-check
pca = PCA(n_components=2)
scores_sklearn = pca.fit_transform(X_pca_std)
print("\nPCA explained variance (manual eigendecomposition):", explained_var_manual[:2].round(3))
print("PCA explained variance (sklearn):", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(scores_sklearn[df.MADE_TOURNEY == 0, 0], scores_sklearn[df.MADE_TOURNEY == 0, 1],
           s=16, color=GRAY, alpha=0.4, label="Did not make tournament")
sc = ax.scatter(scores_sklearn[df.MADE_TOURNEY == 1, 0], scores_sklearn[df.MADE_TOURNEY == 1, 1],
                 s=30, c=df.loc[df.MADE_TOURNEY == 1, "Tour_Wins"], cmap="YlOrRd",
                 edgecolor="black", linewidth=0.3, label="Made tournament")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance) — overall team quality")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance) — playing style")
ax.set_title("Team Seasons Projected onto Two Principal Components")
fig.colorbar(sc, ax=ax, label="Tournament wins")
ax.legend(loc="upper left", frameon=True, fontsize=10)
fig.savefig(f"{FIG_DIR}/05_pca_scatter.png")
plt.close(fig)

# top loadings on PC1/PC2 for interpretation
loadings = pd.DataFrame(pca.components_.T, index=pca_vars, columns=["PC1", "PC2"])
print("\nPCA loadings:\n", loadings.round(2))

print("\nSection 3 (EDA) figures saved to ./figures/")

# %% Section 4: Train / test split (temporal) =================================
# We train on 2013-2018 and hold out the full 2019 season as a genuine
# out-of-sample forecast -- this mimics using the model to project a season
# the group has not yet seen, rather than a random shuffle that could blend
# information across a team's consecutive seasons.
train = df[df.YEAR <= 2018].copy()
test  = df[df.YEAR == 2019].copy()
print(f"Train: {len(train)} team-seasons (2013-2018) | Test: {len(test)} team-seasons (2019)")
print(f"Train tourney rate: {train.MADE_TOURNEY.mean():.1%} | Test tourney rate: {test.MADE_TOURNEY.mean():.1%}")

# %% Section 5: MODEL 1 -- Multiple Linear Regression (Wins ~ Four Factors) ===
# Target: W (season wins, includes conference + NCAA tournament games)
# Features: the eight offense/defense "Four Factors" (Dean Oliver) + tempo

X_train1 = train[FOUR_FACTORS].values
y_train1 = train["W"].values
X_test1  = test[FOUR_FACTORS].values
y_test1  = test["W"].values

# --- 5.1 OLS solved from first principles via the Normal Equations ----------
#     beta_hat = (X'X)^(-1) X'y   ... the closed-form least-squares solution
n, k = X_train1.shape
Xd = np.column_stack([np.ones(n), X_train1])
XtX_inv = np.linalg.inv(Xd.T @ Xd)
beta = XtX_inv @ Xd.T @ y_train1

y_hat_train = Xd @ beta
resid_train = y_train1 - y_hat_train
dof = n - (k + 1)
sigma2 = (resid_train @ resid_train) / dof
se = np.sqrt(np.diag(sigma2 * XtX_inv))
t_stat = beta / se
p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

ss_res = (resid_train ** 2).sum()
ss_tot = ((y_train1 - y_train1.mean()) ** 2).sum()
r2_m1 = 1 - ss_res / ss_tot
adj_r2_m1 = 1 - (1 - r2_m1) * (n - 1) / (n - k - 1)

coef_names = ["Intercept"] + FOUR_FACTORS
model1_table = pd.DataFrame({"coef": beta, "std_err": se, "t_stat": t_stat, "p_value": p_val},
                             index=coef_names)
print("\n=== MODEL 1: Linear Regression coefficient table (train 2013-2018) ===")
print(model1_table.round(4))
print(f"R^2 = {r2_m1:.4f}   Adjusted R^2 = {adj_r2_m1:.4f}   n = {n}")

# cross-check against sklearn (should match to numerical precision)
sk_lr = LinearRegression().fit(X_train1, y_train1)
assert np.allclose(sk_lr.coef_, beta[1:], atol=1e-6), "Manual OLS and sklearn disagree!"
print("[Check] Manual normal-equations solution matches sklearn LinearRegression exactly.")

# --- 5.2 Out-of-sample forecast: 2019 season -------------------------------
Xd_test = np.column_stack([np.ones(len(test)), X_test1])
y_hat_test = Xd_test @ beta
rmse_m1 = np.sqrt(mean_squared_error(y_test1, y_hat_test))
mae_m1  = mean_absolute_error(y_test1, y_hat_test)
r2_test_m1 = r2_score(y_test1, y_hat_test)
print(f"\n2019 HOLDOUT FORECAST -> RMSE = {rmse_m1:.2f} wins, MAE = {mae_m1:.2f} wins, R^2 = {r2_test_m1:.3f}")

# --- 5.3 Diagnostics: actual vs. predicted, residuals ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(y_train1, y_hat_train, s=14, color=NAVY, alpha=0.4, label="Train (2013-18)")
axes[0].scatter(y_test1, y_hat_test, s=22, color=ORANGE, alpha=0.8, label="Test (2019 forecast)")
lims = [0, 40]
axes[0].plot(lims, lims, color=GRAY, linestyle="--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual season wins"); axes[0].set_ylabel("Predicted season wins")
axes[0].set_title(f"Model 1: Actual vs. Predicted Wins\nTrain R²={r2_m1:.3f}  |  2019 Forecast R²={r2_test_m1:.3f}")
axes[0].legend(frameon=False, fontsize=10)

axes[1].scatter(y_hat_train, resid_train, s=14, color=NAVY, alpha=0.4)
axes[1].axhline(0, color=GRAY, linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted season wins"); axes[1].set_ylabel("Residual (actual - predicted)")
axes[1].set_title("Residual Plot (Train)\nNo major pattern = linear model is a reasonable fit")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/06_model1_diagnostics.png")
plt.close(fig)

# --- 5.4 Variance Inflation Factors (multicollinearity check) --------------
vifs = {}
for col in FOUR_FACTORS:
    others = [c for c in FOUR_FACTORS if c != col]
    Xo = np.column_stack([np.ones(n), train[others].values])
    b_o = np.linalg.inv(Xo.T @ Xo) @ Xo.T @ train[col].values
    r2_o = r2_score(train[col].values, Xo @ b_o)
    vifs[col] = 1 / (1 - r2_o)
print("\nVariance Inflation Factors (all comfortably < 5 -> no multicollinearity concern):")
for c, v in sorted(vifs.items(), key=lambda x: -x[1]):
    print(f"  {c}: {v:.2f}")

# %% Section 6: MODEL 2 -- Logistic Regression (Made NCAA Tournament?) =======
# Target: MADE_TOURNEY (binary). Features: opponent-adjusted power metrics,
# since tournament SELECTION depends heavily on strength of schedule, which
# ADJOE/ADJDE/BARTHAG capture (they are opponent-adjusted) but the raw
# Four Factors above do not.

scaler2 = StandardScaler().fit(train[POWER_FEATS])
X_train2 = scaler2.transform(train[POWER_FEATS]); y_train2 = train["MADE_TOURNEY"].values
X_test2  = scaler2.transform(test[POWER_FEATS]);  y_test2  = test["MADE_TOURNEY"].values

logit2 = LogisticRegression().fit(X_train2, y_train2)
prob_test2 = logit2.predict_proba(X_test2)[:, 1]
pred_test2 = logit2.predict(X_test2)

acc2  = accuracy_score(y_test2, pred_test2)
prec2 = precision_score(y_test2, pred_test2)
rec2  = recall_score(y_test2, pred_test2)
f1_2  = f1_score(y_test2, pred_test2)
auc2  = roc_auc_score(y_test2, prob_test2)
brier2 = brier_score_loss(y_test2, prob_test2)
cm2 = confusion_matrix(y_test2, pred_test2)

print("\n=== MODEL 2: Logistic Regression -- Made NCAA Tournament? (2019 forecast) ===")
print("Standardized coefficients (log-odds per +1 SD) and odds ratios:")
for name, c in zip(POWER_FEATS, logit2.coef_[0]):
    print(f"  {name:8s}: beta={c:+.3f}   odds x{np.exp(c):.2f} per +1 SD")
print(f"Accuracy={acc2:.3f}  Precision={prec2:.3f}  Recall={rec2:.3f}  F1={f1_2:.3f}  AUC={auc2:.3f}  Brier={brier2:.4f}")
print(f"Baseline (always predict 'missed tournament'): {1 - y_test2.mean():.3f} accuracy")
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm2)

# --- 6.1 ROC curve + confusion matrix figure --------------------------------
fpr, tpr, _ = roc_curve(y_test2, prob_test2)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].plot(fpr, tpr, color=NAVY, linewidth=2.5, label=f"Model (AUC = {auc2:.3f})")
axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle="--", label="Random guessing (AUC = 0.50)")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Model 2 ROC Curve\nMade NCAA Tournament (2019 forecast)")
axes[0].legend(loc="lower right", frameon=False)

im = axes[1].imshow(cm2, cmap="Blues")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Predicted: Missed", "Predicted: Made"])
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Actual: Missed", "Actual: Made"])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm2[i, j], ha="center", va="center",
                      fontsize=16, color="white" if cm2[i, j] > cm2.max() / 2 else "black")
axes[1].set_title("Confusion Matrix (2019, n=353)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/07_model2_roc_confusion.png")
plt.close(fig)

# --- 6.2 Illustrative team-level predictions (for the forecast slide) ------
test_disp2 = test.copy()
test_disp2["PRED_PROB"] = prob_test2
bubble = test_disp2[(test_disp2.PRED_PROB >= 0.30) & (test_disp2.PRED_PROB <= 0.70)]
bubble = bubble.sort_values("PRED_PROB", ascending=False)[
    ["TEAM", "CONF", "W", "BARTHAG", "PRED_PROB", "MADE_TOURNEY", "POSTSEASON"]]
bubble = bubble.round({"PRED_PROB": 3})
bubble.to_csv(f"{FIG_DIR}/../bubble_teams_2019_forecast.csv", index=False)
print(f"\n{len(bubble)} 'bubble zone' teams (predicted prob 30-70%) saved to bubble_teams_2019_forecast.csv")
print(bubble.head(12).to_string(index=False))

# %% Section 7: EXTENSION -- Random Forest for "how far will they go?" ========
# Restrict the universe to the 68 teams that make the tournament each year,
# and ask: will this team reach the Sweet 16 or better (Tour_Wins >= 2)?
# We compare a Random Forest against (a) a logistic regression on the same
# features and (b) a "seed-only" baseline representing the committee's own
# seeding -- to see whether our stats-based model can match/beat the
# committee's judgment using only regular-season performance metrics.

tdf = df[df.MADE_TOURNEY == 1].copy()
t_train = tdf[tdf.YEAR <= 2018].copy()
t_test  = tdf[tdf.YEAR == 2019].copy()

scaler3 = StandardScaler().fit(t_train[EXT_FEATS])
X_train3 = scaler3.transform(t_train[EXT_FEATS]); y_train3 = t_train["SWEET16_PLUS"].values
X_test3  = scaler3.transform(t_test[EXT_FEATS]);  y_test3  = t_test["SWEET16_PLUS"].values

logit3 = LogisticRegression().fit(X_train3, y_train3)
prob3_logit = logit3.predict_proba(X_test3)[:, 1]

rf3 = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=4, random_state=42)
rf3.fit(X_train3, y_train3)
prob3_rf = rf3.predict_proba(X_test3)[:, 1]

seed_logit = LogisticRegression().fit(t_train[["SEED"]], t_train["SWEET16_PLUS"])
prob3_seed = seed_logit.predict_proba(t_test[["SEED"]])[:, 1]

print("\n=== EXTENSION: Sweet-16-or-better among the 68 tournament teams (2019) ===")
for label, prob in [("Logistic Regression", prob3_logit), ("Random Forest", prob3_rf), ("Seed-only baseline", prob3_seed)]:
    acc = accuracy_score(y_test3, (prob >= 0.5).astype(int))
    auc = roc_auc_score(y_test3, prob)
    brier = brier_score_loss(y_test3, prob)
    print(f"  {label:22s}: Accuracy={acc:.3f}  AUC={auc:.3f}  Brier={brier:.4f}")

importances = pd.Series(rf3.feature_importances_, index=EXT_FEATS).sort_values(ascending=False)
print("\nRandom Forest feature importances:\n", importances.round(3))

# --- 7.1 Feature importance + ROC comparison figure -------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].barh(importances.index[::-1], importances.values[::-1], color=NAVY)
axes[0].set_xlabel("Relative importance")
axes[0].set_title("What Drives a Deep Tournament Run?\n(Random Forest feature importance)")

for label, prob, color in [("Our model (Random Forest)", prob3_rf, NAVY),
                            ("Logistic regression", prob3_logit, ORANGE),
                            ("Seed only (committee)", prob3_seed, GRAY)]:
    fpr_, tpr_, _ = roc_curve(y_test3, prob)
    auc_ = roc_auc_score(y_test3, prob)
    axes[1].plot(fpr_, tpr_, linewidth=2.2, color=color, label=f"{label} (AUC={auc_:.3f})")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.3)
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Reaching the Sweet 16: Our Model vs.\nthe Committee's Own Seed")
axes[1].legend(loc="lower right", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/08_extension_rf_vs_seed.png")
plt.close(fig)

# --- 7.2 Full 2019 tournament-team forecast table (for the "Oregon" story) -
t_test_disp = t_test.copy()
t_test_disp["PRED_PROB_RF"] = prob3_rf
t_test_disp = t_test_disp.sort_values("PRED_PROB_RF", ascending=False)
t_test_out = t_test_disp[["TEAM", "SEED", "WAB", "PRED_PROB_RF", "SWEET16_PLUS", "POSTSEASON"]].round({"PRED_PROB_RF": 3})
t_test_out.to_csv(f"{FIG_DIR}/../sweet16_forecast_2019_full.csv", index=False)
biggest_miss = (t_test_disp["SWEET16_PLUS"] - t_test_disp["PRED_PROB_RF"]).abs().idxmax()
print("\nBiggest single forecast miss (2019):")
print(t_test_disp.loc[[biggest_miss], ["TEAM", "SEED", "WAB", "PRED_PROB_RF", "SWEET16_PLUS", "POSTSEASON"]])

# %% Section 8: FORECAST SUMMARY (consolidated, for the presentation) =========
print("\n" + "=" * 70)
print("FORECAST SUMMARY -- 2019 season treated as unseen/future data")
print("=" * 70)
print(f"Model 1 (Wins, linear regression):        RMSE = {rmse_m1:.2f} wins, R^2 = {r2_test_m1:.3f}")
print(f"Model 2 (Made Tournament, logistic):       Accuracy = {acc2:.3f}, AUC = {auc2:.3f}, Brier = {brier2:.4f}")
print(f"Extension (Sweet16+, random forest):       Accuracy = {accuracy_score(y_test3,(prob3_rf>=0.5).astype(int)):.3f}, "
      f"AUC = {roc_auc_score(y_test3, prob3_rf):.3f}")
print("=" * 70)

# %% Section 9: Export a compact results summary for use while building slides
with open(f"{FIG_DIR}/../results_summary.txt", "w") as f:
    f.write("MODEL 1 -- Linear Regression (Wins ~ Four Factors)\n")
    f.write(model1_table.round(4).to_string())
    f.write(f"\nR^2={r2_m1:.4f}  Adj R^2={adj_r2_m1:.4f}\n")
    f.write(f"2019 forecast: RMSE={rmse_m1:.2f} wins  MAE={mae_m1:.2f} wins  R^2={r2_test_m1:.3f}\n\n")

    f.write("MODEL 2 -- Logistic Regression (Made Tournament ~ Efficiency)\n")
    for name, c in zip(POWER_FEATS, logit2.coef_[0]):
        f.write(f"  {name}: beta={c:+.3f}  odds_ratio={np.exp(c):.2f}\n")
    f.write(f"2019 forecast: Accuracy={acc2:.3f} Precision={prec2:.3f} Recall={rec2:.3f} "
            f"F1={f1_2:.3f} AUC={auc2:.3f} Brier={brier2:.4f}\n\n")

    f.write("EXTENSION -- Random Forest (Sweet16+ among tourney teams)\n")
    f.write(importances.round(3).to_string())
    f.write(f"\n2019 forecast: Accuracy={accuracy_score(y_test3,(prob3_rf>=0.5).astype(int)):.3f} "
            f"AUC={roc_auc_score(y_test3, prob3_rf):.3f}\n")
    f.write(f"Seed-only baseline: Accuracy={accuracy_score(y_test3,(prob3_seed>=0.5).astype(int)):.3f} "
            f"AUC={roc_auc_score(y_test3, prob3_seed):.3f}\n")

print("\nAll figures saved in ./figures/, tables saved as .csv, and a plain-text\n"
      "results_summary.txt was written for quick reference while building slides.")

