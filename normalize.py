import pandas as pd
import numpy as np

def normalize_efd_dataset(input_filepath, output_filepath):
    """
    Normalizes Elliptical Fourier Descriptor (EFD) coefficients to be invariant
    to size using the correct matrix transformation method.

    This function reads a CSV file of EFD coefficients, calculates the semi-major
    axis (p) from the first harmonic for each sample, and then divides all
    coefficients for that sample by its corresponding p-value.

    The corrected normalization process for each sample (row) is as follows:
    1. A 2x2 transformation matrix T is constructed from the first harmonic
       coefficients (a1, b1, c1, d1).
    2. A new matrix M is calculated as M = T * T_transpose.
    3. Eigen decomposition is performed on M.
    4. The largest eigenvalue of M (lambda_max) is found.
    5. The semi-major axis is calculated as p = sqrt(lambda_max).
    6. All 40 harmonic coefficients for the sample are divided by p.

    Args:
        input_filepath (str): The path to the input CSV file with EFD coefficients.
        output_filepath (str): The path where the normalized CSV file will be saved.
    """
    print(f"Reading data from '{input_filepath}'...")
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return

    # Keep a copy of the original data for metadata
    df_normalized = df.copy()

    print("Normalizing the dataset with corrected logic...")

    # --- Corrected Vectorized Calculation of Semi-Major Axis (p) ---

    # 1. Construct a stack of 2x2 transformation matrices (T) from the first harmonic.
    # The shape will be (number_of_samples, 2, 2)
    T_matrices = np.array([
        df['a1'], df['b1'],
        df['c1'], df['d1']
    ]).T.reshape(-1, 2, 2)

    # 2. Get the transpose of each matrix in the stack.
    T_transpose_matrices = np.transpose(T_matrices, (0, 2, 1))

    # 3. Compute M = T * T_transpose for each matrix in the stack.
    # The @ operator performs matrix multiplication.
    M_matrices = T_matrices @ T_transpose_matrices

    # 4. Calculate the eigenvalues for all M matrices at once.
    eigenvalues = np.linalg.eigvals(M_matrices)

    # 5. Find the largest eigenvalue (lambda_max) for each sample.
    lambda_max = np.max(eigenvalues.real, axis=1)

    # 6. Calculate the semi-major axis (p).
    # Use np.abs to handle any potential small negative floats from precision errors.
    p = np.sqrt(np.abs(lambda_max))

    # Avoid division by zero for samples with zero size.
    p[p == 0] = 1

    # --- Normalization ---

    # Get a list of all coefficient column names
    coeff_columns = [f'{coeff}{i}' for coeff in 'abcd' for i in range(1, 11)]
    
    # Ensure all expected columns exist
    missing_cols = [col for col in coeff_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing: {missing_cols}")
        return

    # 7. Divide all coefficient columns by the corresponding p-value for each row.
    df_normalized[coeff_columns] = df[coeff_columns].div(p, axis=0)

    # --- Save the Result ---
    try:
        df_normalized.to_csv(output_filepath, index=False)
        print(f"Successfully normalized the data and saved it to '{output_filepath}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == '__main__':
    input_csv = r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\flip_efd_coefficients_10h.csv"
    output_csv = r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv"
    normalize_efd_dataset(input_csv, output_csv)

