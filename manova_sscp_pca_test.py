import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

# --- Load dataset ---
# NOTE: You will need to replace this path with the actual location of your file.
try:
    df = pd.read_csv(
        r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv"
    )
except FileNotFoundError:
    print("File not found. Please update the file path in the script.")
    # As a fallback for execution, create a dummy dataframe
    data = {'species': ['A', 'A', 'B', 'B'] * 10,
            'gender': ['M', 'F', 'M', 'F'] * 10,
            'image_id': [f'id_{i}' for i in range(40)]}
    for i in range(40):
        data[f'H{i}'] = np.random.rand(40)
    df = pd.DataFrame(data)

# --- Extract harmonics (features) ---
harmonics_cols = [c for c in df.columns if c not in ["image_id", "species", "gender"]]
Y_raw = df[harmonics_cols].values

# --- Encode factors ---
df["species"] = df["species"].astype("category")
df["gender"] = df["gender"].astype("category")

# --- Model setup ---
formula = "species * gender"
full_model = ols(f"{harmonics_cols[0]} ~ {formula}", data=df).fit()
X_full = full_model.model.exog
term_slices = full_model.model.data.design_info.term_name_slices

# --- Helper functions ---
def projection(X):
    return X @ np.linalg.pinv(X.T @ X) @ X.T

def type3_sscp(Y, X_full, term_slices):
    P_full = projection(X_full)
    Y_fullhat = P_full @ Y
    sscp_dict = {}
    for term, sl in term_slices.items():
        if term == "Intercept": continue
        cols_keep = np.setdiff1d(np.arange(X_full.shape[1]), np.arange(sl.start, sl.stop))
        X_reduced = X_full[:, cols_keep]
        P_reduced = projection(X_reduced)
        Y_reducedhat = P_reduced @ Y
        effect = Y_fullhat - Y_reducedhat
        H = effect.T @ effect
        sscp_dict[term] = np.trace(H)
    resid = Y - Y_fullhat
    sscp_dict["Residuals"] = np.trace(resid.T @ resid)
    return sscp_dict

def sscp_percent(sscp_dict):
    total = sum(sscp_dict.values())
    return {k: v / total * 100 for k, v in sscp_dict.items()}

# --- Prepare data ---
Y_std = StandardScaler().fit_transform(Y_raw)
pca = PCA()
Y_pca_full = pca.fit_transform(Y_std)

# --- MODIFICATION: Loop through different numbers of PCs ---
results = []
labels = []

# 1. Baseline: All 40 Standardized Features
sscp_std = type3_sscp(Y_std, X_full, term_slices)
results.append(sscp_percent(sscp_std))
labels.append("All Features (40)")

# 2. PCA-based analyses
pc_counts = [10, 20, 30, 40]
for count in pc_counts:
    # Select the subset of PCs
    Y_pca_subset = Y_pca_full[:, :count]

    # Calculate SSCP and percentages
    sscp_pca = type3_sscp(Y_pca_subset, X_full, term_slices)
    results.append(sscp_percent(sscp_pca))

    # Get variance explained for the label
    variance_explained = np.sum(pca.explained_variance_ratio_[:count])
    # Corrected f-string formatting
    labels.append(f"{count} PCs ({variance_explained:.1%})")

# --- Create comparison DataFrame and Plot ---
df_compare = pd.DataFrame(results, index=labels)
print("\n=== Comparison of SSCP Percentage Contributions ===")
print(df_compare)

# Create grouped bar plot
ax = df_compare.plot(kind='bar', figsize=(14, 8), rot=45)
plt.title("SSCP Contribution vs. Number of Principal Components")
plt.ylabel("Contribution (%)")
plt.xlabel("Analysis Method")
plt.legend(title="Model Term", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("sscp_expanded_comparison.png")

print("\nGenerated expanded comparison plot: sscp_expanded_comparison.png")