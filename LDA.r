# Description: Performs LDA on fly wing harmonic data and saves results for Python plotting.

# --- 1. Install and Load Required Libraries ---
# Run these lines once if you haven't installed the packages yet
# install.packages("readr")
# install.packages("dplyr")
# install.packages("MASS")

library(readr)
library(dplyr)
library(MASS) # Contains the lda() function

# --- 2. Load and Prepare the Data ---
# Assumes the CSV is in the same directory as the script.
input_file <- "normalized_efd_coefficients_10h.csv"

if (!file.exists(input_file)) {
  stop("Error: The input file '", input_file, "' was not found. Please place it in the script's directory.")
}

full_data <- read_csv(input_file, show_col_types = FALSE)

# Standardize 'sex' column name to match previous scripts
if ("gender" %in% names(full_data) && !"sex" %in% names(full_data)) {
  full_data <- rename(full_data, sex = gender)
  print("Renamed column 'gender' to 'sex' for consistency.")
}

# --- 3. Separate Features (X) and Target (y) ---
harmonic_columns <- grep("^[abcd][0-9]+$", names(full_data), value = TRUE)
X <- full_data[harmonic_columns]
y <- full_data$species

# Store metadata for the final output file
# ✅ FIXED: Use dplyr::select to avoid conflict with MASS::select
metadata <- full_data %>% dplyr::select(species, sex)

# --- 4. Scale Features and Perform LDA ---
# LDA benefits from standardized data, same as PCA
X_scaled <- scale(X)

# Perform Linear Discriminant Analysis
# The target variable 'y' must be a factor for lda()
lda_model <- lda(X_scaled, grouping = as.factor(y))

# Predict the LD components for the scaled data
lda_components <- predict(lda_model, X_scaled)$x

# Keep only the first 3 components
lda_df <- as.data.frame(lda_components[, 1:3])
colnames(lda_df) <- c("LD1", "LD2", "LD3")


# --- 5. Combine and Save Results ---
# Combine original metadata with the LDA results
final_df <- bind_cols(metadata, lda_df)

# Save the combined data frame to a CSV file
write_csv(final_df, "lda_results.csv")


# --- 6. Calculate and Save Explained Variance ---
# The proportion of trace is the explained variance for each component
singular_values <- lda_model$svd
explained_variance <- (singular_values^2) / sum(singular_values^2)

variance_df <- data.frame(
  var_LD1 = explained_variance[1],
  var_LD2 = explained_variance[2],
  var_LD3 = explained_variance[3]
)

# Save the variance data to a separate CSV
write_csv(variance_df, "lda_variance.csv")


# --- 7. Confirmation Message ---
print("✅ LDA calculation complete. Two files were generated:")
print("- lda_results.csv (LD scores with species and sex)")

print("- lda_variance.csv (Explained variance for each LD)")
