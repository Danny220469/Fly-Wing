# 1. Load Libraries
library(Momocs)
library(imager)

# 2. Set Up Your Folders
image_folder <- "C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/Image"
output_file <- "C:/Users/User/Documents/Bioinformatics_Year3_Sem2/Internship/Fly Project/Image/test_efd_coefficients_10h.csv"

# Helper function to extract largest contour by area
get_largest_contour <- function(contours) {
    if (length(contours) == 0) return(NULL)
    contour_areas <- sapply(contours, function(contour) {
        x <- contour$x
        y <- contour$y
        abs(sum(x[-1] * y[-length(y)] - x[-length(x)] * y[-1]) / 2)
    })
    contours[[which.max(contour_areas)]]
}

harmonics <- 10
all_results <- list()

# List all subfolders in image_folder
subfolders <- list.dirs(image_folder, full.names = TRUE, recursive = TRUE)

for (subfolder in subfolders) {
    sam_folder <- file.path(subfolder, "SAM")
    if (dir.exists(sam_folder)) {
        # Get species name by removing "Female - " prefix
        species_name <- basename(subfolder)
        species_name <- sub("^Female - ", "", species_name)
        mask_files <- list.files(path = sam_folder, pattern = "\\.png$", full.names = TRUE)
        for (file_path in mask_files) {
            filename <- basename(file_path)
            cat("Processing:", filename, "in species:", species_name, "\n")
            img <- load.image(file_path)
            img <- mirror(img, "y")  # Flip image if necessary
            contour_list <- imager::contours(img)
            if (!is.null(contour_list) && length(contour_list) > 0) {
                largest_contour <- get_largest_contour(contour_list)
                contour_matrix <- cbind(largest_contour$x, largest_contour$y)
                if (nrow(contour_matrix) > 5) {
                    nb.h <- min(harmonics, nrow(contour_matrix) %/% 2)
                    tryCatch({
                        coe <- efourier(contour_matrix, nb.h = nb.h, norm = FALSE)
                        actual_harmonics <- length(coe$an)
                        coef_data <- data.frame(
                            image_id = filename,
                            species = species_name,
                            t(c(coe$an, coe$bn, coe$cn, coe$dn))
                        )
                        colnames(coef_data)[3:(2 + 4 * actual_harmonics)] <- unlist(lapply(c("a", "b", "c", "d"), function(prefix) paste0(prefix, 1:actual_harmonics)))
                        all_results[[paste0(species_name, "_", filename)]] <- coef_data
                    }, error = function(e) {
                        cat("   Error in efourier for:", filename, "-", e$message, "\n")
                    })
                } else {
                    cat("   Insufficient points in largest contour for:", filename, "\n")
                }
            } else {
                cat("   Failed to extract contours for:", filename, "\n")
            }
        }
    }
}

# Combine all results and write to CSV
if (length(all_results) > 0) {
    efd_df <- do.call(rbind, all_results)
    write.csv(efd_df, file = output_file, row.names = FALSE)
    cat("EFD complete! Coefficients for", length(all_results), "images saved to", output_file, "\n")
} else {
    cat("No valid coefficients produced.\n")
}