# Description: Performs PCA on fly wing harmonic data and saves results for Python plotting.

# --- 1. Install and Load Required Libraries ---
# Run these lines once if you haven't installed the packages yet
# install.packages("readr")
# install.packages("dplyr")

library(readr)
library(dplyr)

# --- 2. Load the Data ---
# Assumes the CSV is in the same directory as the script.
input_file <- "normalized_efd_coefficients_10h.csv"

if (!file.exists(input_file)) {
  stop("Error: The input file '", input_file, "' was not found. Please place it in the script's directory.")
}

full_data <- read_csv(input_file, show_col_types = FALSE)

# --- 3. Prepare Data for PCA ---
# Isolate the harmonic coefficient columns (a1-a10, b1-b10, etc.)
harmonic_columns <- grep("^[abcd][0-9]+$", names(full_data), value = TRUE)
X <- full_data[harmonic_columns]

# Keep metadata for later
metadata <- full_data %>% dplyr::select(species, sex)

# --- 4. Perform PCA ---
# prcomp with center=TRUE and scale.=TRUE is equivalent to Scikit-learn's
# StandardScaler() followed by PCA().
pca_result <- prcomp(X, center = TRUE, scale. = TRUE)

# Extract the first 3 principal components
pcs <- as.data.frame(pca_result$x[, 1:3])
colnames(pcs) <- c("PC1", "PC2", "PC3")

# --- 5. Combine and Save Results ---
# Combine metadata with the PCA results
final_df <- bind_cols(metadata, pcs)

# Save the combined data frame to a CSV file
write_csv(final_df, "pca_results.csv")

# --- 6. Save Explained Variance ---
# Get the summary and extract the "Proportion of Variance" for the first 3 PCs
explained_variance <- summary(pca_result)$importance[2, 1:3]
variance_df <- data.frame(
  var_PC1 = explained_variance[1],
  var_PC2 = explained_variance[2],
  var_PC3 = explained_variance[3]
)

# Save the variance data to a separate CSV
write_csv(variance_df, "pca_variance.csv")

# --- 7. Confirmation Message ---
print("✅ PCA calculation complete. Two files were generated:")
print("- pca_results.csv (PC scores with species and sex)")
print("- pca_variance.csv (Explained variance for each PC)")