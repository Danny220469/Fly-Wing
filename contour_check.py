import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Load the CSV ===
file_path = r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv"
df = pd.read_csv(file_path)

# === Function: reconstruct contour from EFD coefficients ===
def reconstruct_contour(coeffs, num_points=300):
    n_harmonics = 10
    t = np.linspace(0, 2 * np.pi, num_points)
    xt = np.zeros(num_points)
    yt = np.zeros(num_points)

    for n in range(1, n_harmonics + 1):
        an, bn = coeffs.get(f'a{n}', 0), coeffs.get(f'b{n}', 0)
        cn, dn = coeffs.get(f'c{n}', 0), coeffs.get(f'd{n}', 0)
        xt += an * np.cos(n * t) + bn * np.sin(n * t)
        yt += cn * np.cos(n * t) + dn * np.sin(n * t)

    return xt, yt

# === Species list ===
species_list = df['species'].unique()

# === Create subplots (2x4 for 8 species) ===
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()
fig.suptitle('Male vs Female Wing Contours by Species (Reconstructed from EFD)', fontsize=16)

for i, species_name in enumerate(species_list):
    if i >= len(axes):
        break
    ax = axes[i]
    species_df = df[df['species'] == species_name]

    all_contours = {"male": {"x": [], "y": []}, "female": {"x": [], "y": []}}

    # --- plot all individual contours in light grey ---
    for _, row in species_df.iterrows():
        coeffs = row.to_dict()
        x, y = reconstruct_contour(coeffs)

        ax.plot(x, y, color="grey", alpha=0.2, linewidth=1)  # << grey for individuals

        if row["gender"] == "female":
            all_contours["female"]["x"].append(x)
            all_contours["female"]["y"].append(y)
        elif row["gender"] == "male":
            all_contours["male"]["x"].append(x)
            all_contours["male"]["y"].append(y)

    # --- compute and plot mean contour for each gender ---
    if all_contours["female"]["x"]:
        mean_x = np.mean(all_contours["female"]["x"], axis=0)
        mean_y = np.mean(all_contours["female"]["y"], axis=0)
        ax.plot(mean_x, mean_y, color="red", linewidth=2.5, linestyle="-", label="Female Mean")

    if all_contours["male"]["x"]:
        mean_x = np.mean(all_contours["male"]["x"], axis=0)
        mean_y = np.mean(all_contours["male"]["y"], axis=0)
        ax.plot(mean_x, mean_y, color="blue", linewidth=2.5, linestyle="--", label="Male Mean")

    ax.set_title(species_name, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# === Save ===
output_dir = "contour_plots"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "male_vs_female_wing_contours_by_species.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved species-wise contour plots (male vs female) to {output_path}")
