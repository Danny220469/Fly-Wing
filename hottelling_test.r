# --- Load required packages ---
library(tidyverse)
library(MASS)

mahalanobis_hotelling_pca <- function(df, harmonics_pattern = "^[abcd][0-9]+$", 
                                      var_threshold = 0.9,
                                      male_label = "male", female_label = "female") {
  
  # --- Identify harmonics and run PCA ---
  harmonics <- grep(harmonics_pattern, names(df), value = TRUE)
  harmonics_mat <- scale(df[, harmonics])
  
  pca <- prcomp(harmonics_mat, center = TRUE, scale. = TRUE)
  var_explained <- summary(pca)$importance[3, ]
  n_pc <- which(var_explained >= var_threshold)[1]
  cat("Number of PCs explaining >", var_threshold * 100, "% variance:", n_pc, "\n")
  
  pc_scores <- pca$x[, 1:n_pc, drop = FALSE]
  pc_names <- paste0("PC", 1:n_pc)
  df_pca <- cbind(df[, c("species", "gender")], as.data.frame(pc_scores))
  colnames(df_pca)[-(1:2)] <- pc_names
  
  # --- Function to compute Mahalanobis distance ---
  compute_mahalanobis <- function(mat, labels) {
    male_mat <- mat[labels == male_label, , drop = FALSE]
    female_mat <- mat[labels == female_label, , drop = FALSE]
    if (nrow(male_mat) < 2 || nrow(female_mat) < 2) return(NA_real_)
    
    mean_male <- colMeans(male_mat)
    mean_female <- colMeans(female_mat)
    cov_male <- cov(male_mat)
    cov_female <- cov(female_mat)
    pooled_cov <- ((nrow(male_mat) - 1) * cov_male + (nrow(female_mat) - 1) * cov_female) /
      (nrow(male_mat) + nrow(female_mat) - 2)
    inv_cov <- tryCatch(solve(pooled_cov), error = function(e) MASS::ginv(pooled_cov))
    diff <- mean_male - mean_female
    dist_sq <- as.numeric(t(diff) %*% inv_cov %*% diff)
    sqrt(dist_sq)
  }
  
  # --- Run Hotelling test per species ---
  species_list <- unique(df_pca$species)
  results <- lapply(species_list, function(sp) {
    subdf <- df_pca %>% filter(species == sp)
    mat <- as.matrix(subdf[, pc_names])
    labels <- as.character(subdf$gender)
    
    male <- mat[labels == male_label, , drop = FALSE]
    female <- mat[labels == female_label, , drop = FALSE]
    
    if (nrow(male) < 2 || nrow(female) < 2) {
      return(tibble(species = sp, mahalanobis_dist = NA_real_,
                    T2 = NA_real_, Fstat = NA_real_, df1 = NA_real_, df2 = NA_real_,
                    pvalue = NA_real_))
    }
    
    # Mahalanobis distance
    mahalanobis_dist <- compute_mahalanobis(mat, labels)
    
    # Sample sizes
    n_x <- nrow(male)
    n_y <- nrow(female)
    p <- ncol(male)  # number of PCs
    
    # Hotelling's T2 (manual calculation)
    mean_male <- colMeans(male)
    mean_female <- colMeans(female)
    cov_male <- cov(male)
    cov_female <- cov(female)
    pooled_cov <- ((n_x - 1) * cov_male + (n_y - 1) * cov_female) / (n_x + n_y - 2)
    inv_cov <- tryCatch(solve(pooled_cov), error = function(e) MASS::ginv(pooled_cov))
    diff <- mean_male - mean_female
    T2_val <- as.numeric((n_x * n_y) / (n_x + n_y) * t(diff) %*% inv_cov %*% diff)
    
    # Compute F statistic
    F_val <- ((n_x + n_y - p - 1) / ((n_x + n_y - 2) * p)) * T2_val
    
    # Degrees of freedom
    df1_val <- p
    df2_val <- n_x + n_y - p - 1
    
    # p-value
    p_val <- 1 - pf(F_val, df1_val, df2_val)
    
    tibble(
      species = sp,
      mahalanobis_dist = mahalanobis_dist,
      T2 = T2_val,
      Fstat = F_val,
      df1 = df1_val,
      df2 = df2_val,
      pvalue = p_val
    )
  })
  
  results_df <- bind_rows(results) %>% arrange(desc(mahalanobis_dist))
  return(results_df)
}

# --- Example usage ---
file_path <- "C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/normalized_efd_coefficients_10h.csv"
df <- read.csv(file_path)

results <- mahalanobis_hotelling_pca(df, var_threshold = 0.9)
print(results)
