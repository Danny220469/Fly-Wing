# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# --- 1. Load and Prepare the Data ---
tryCatch({
    df <- read_csv("C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/normalized_efd_coefficients_10h.csv")
}, error = function(e) {
    stop("File not found. Please update the file path in the script.")
})


# --- Extract harmonics and prepare data matrices ---
harmonics_cols <- setdiff(names(df), c("image_id", "species", "sex"))
Y_raw <- as.matrix(df[, harmonics_cols])

# --- Encode factors and add the family column ---
df <- df %>%
    mutate(
        species = as.factor(species),
        sex = as.factor(sex),
        family = as.factor(case_when(
            species == "Synthesiomyia nudiseta" ~ "Muscidae",
            TRUE ~ "Calliphoridae"
        ))
    )

# --- Helper function to calculate SSCP percentage contribution ---
calculate_sscp_percent <- function(dependent_vars, factors) {
    model_data <- cbind(as.data.frame(dependent_vars), factors)
    
    # --- MODEL 1 FORMULA ---
    # This formula focuses on the family:sex interaction.
    formula_str <- paste("cbind(", paste(colnames(dependent_vars), collapse = ","), ") ~ family + species + sex + family:sex")
    
    manova_fit <- manova(as.formula(formula_str), data = model_data)
    sscp_matrices <- summary(manova_fit)$SS
    matrix_trace <- function(mat) { sum(diag(mat)) }
    sscp_traces <- sapply(sscp_matrices, matrix_trace)
    total_sscp <- sum(sscp_traces)
    sscp_percent <- (sscp_traces / total_sscp) * 100
    
    return(sscp_percent)
}


# --- Prepare data for analysis ---
Y_std <- scale(Y_raw, center = TRUE, scale = TRUE)
pca_result <- prcomp(Y_std, scale. = FALSE, center = FALSE) 
Y_pca_full <- pca_result$x

# --- Loop through different numbers of PCs to get results ---
results_list <- list()
factor_cols <- c("species", "sex", "family")

# Baseline analysis
results_list[["All Features (40)"]] <- calculate_sscp_percent(Y_std, df[, factor_cols])

# PCA-based analyses
pc_counts <- c(10, 20, 30, 40)
variance_explained_ratios <- summary(pca_result)$importance["Cumulative Proportion", ]
for (count in pc_counts) {
    Y_pca_subset <- Y_pca_full[, 1:count, drop = FALSE]
    sscp_pca <- calculate_sscp_percent(Y_pca_subset, df[, factor_cols])
    label <- paste0(count, " PCs (", percent(variance_explained_ratios[count], accuracy = 0.1), ")")
    results_list[[label]] <- sscp_pca
}


# --- Create comparison DataFrame and Plot with ggplot2 ---
df_compare <- as.data.frame(do.call(rbind, results_list))
df_compare$analysis <- rownames(df_compare)

print("=== Comparison of SSCP Percentage Contributions (with Family:Sex) ===")
print(df_compare)

df_long <- df_compare %>%
    pivot_longer(cols = -analysis, names_to = "term", values_to = "contribution") %>%
    mutate(analysis = factor(analysis, levels = names(results_list)))

plot <- ggplot(df_long, aes(x = analysis, y = contribution, fill = term)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    labs(
        title = "SSCP Contribution (Model with family:sex Interaction)",
        y = "Contribution (%)",
        x = "Analysis Method",
        fill = "Model Term"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "right"
    )

# Save the plot to a unique file
ggsave("sscp_comparison_family_sex_interaction.png", plot = plot, width = 14, height = 8, units = "in", dpi = 300)

print("✅ Generated plot 1: sscp_comparison_family_sex_interaction.png")