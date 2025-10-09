# Title: normalize.R
# Author: Gemini (Translated from Python)
# Date: 2025-10-09
# Description: Normalizes Elliptical Fourier Descriptor (EFD) coefficients to be invariant to size.

# --- 1. Install and Load Required Libraries ---
# Run these lines once if you haven't installed the packages yet
# install.packages("readr")
# install.packages("dplyr")

library(readr)
library(dplyr)


normalize_efd_dataset_r <- function(input_filepath, output_filepath) {
  #' Normalizes Elliptical Fourier Descriptor (EFD) coefficients.
  #'
  #' This function reads a CSV file of EFD coefficients, calculates the semi-major
  #' axis (p) from the first harmonic for each sample, and then divides all
  #' coefficients for that sample by its corresponding p-value.
  #'
  #' @param input_filepath The path to the input CSV file.
  #' @param output_filepath The path where the normalized CSV file will be saved.
  
  cat(sprintf("Reading data from '%s'...\n", input_filepath))
  tryCatch({
    df <- readr::read_csv(input_filepath, show_col_types = FALSE)
  }, error = function(e) {
    stop(sprintf("Error: The file '%s' was not found or could not be read.", input_filepath))
  })
  
  # Keep a copy of the original data
  df_normalized <- df
  
  cat("Normalizing the dataset...\n")
  
  # --- Calculation of Semi-Major Axis (p) ---
  
  # ✅ FIXED: Only pass the required numeric columns to apply()
  # This prevents R from converting numbers to text.
  p_values <- apply(df[, c("a1", "b1", "c1", "d1")], 1, function(row) {
    # 1. Construct the 2x2 transformation matrix T from the first harmonic.
    T_matrix <- matrix(c(row["a1"], row["c1"], row["b1"], row["d1"]), nrow = 2)
    
    # 2. Compute M = T * T_transpose.
    M_matrix <- T_matrix %*% t(T_matrix)
    
    # 3. & 4. Calculate eigenvalues of M and find the largest one.
    eigenvalues <- eigen(M_matrix, only.values = TRUE)$values
    lambda_max <- max(eigenvalues)
    
    # 5. Calculate the semi-major axis (p).
    p <- sqrt(abs(lambda_max))
    
    return(p)
  })
  
  # Avoid division by zero for samples with zero size.
  p_values[p_values == 0] <- 1
  
  # --- Normalization ---
  
  # Get a list of all coefficient column names using regex
  coeff_columns <- grep("^[abcd][0-9]+$", names(df), value = TRUE)
  
  if (length(coeff_columns) < 40) {
      warning("Warning: Fewer than 40 harmonic coefficient columns (a1-d10) were found.")
  }

  # 6. Divide all coefficient columns by the corresponding p-value for each row.
  df_normalized[coeff_columns] <- df[coeff_columns] / p_values
  
  # --- Save the Result ---
  tryCatch({
    readr::write_csv(df_normalized, output_filepath)
    cat(sprintf("Successfully normalized the data and saved it to '%s'\n", output_filepath))
  }, error = function(e) {
    stop(sprintf("An error occurred while saving the file: %s", e$message))
  })
}

# --- Main execution block ---

# Define file paths
input_csv <- "C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/flip_efd_coefficients_10h.csv"
output_csv <- "C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/normalized_efd_coefficients_10h.csv"

# Run the normalization function
normalize_efd_dataset_r(input_csv, output_csv)