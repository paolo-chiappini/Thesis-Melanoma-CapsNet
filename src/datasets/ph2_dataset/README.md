# PH2 Dataset

## Preprocessing

The original data, saved in a xlsx file cannot be parsed by pandas due to the presence of a legend table.
The first step was to remove the legend, resulting in the `PH2_dataset_reformatted.xlsx` file.
The second step involves encoding values into a one-hot-ecoded state. For binary attributes, this is straightforward: Xs indicating the presence of said attribute were replaced with 1s and NaNs with 0s. For multi-valued attributes, OHE was applied preserving the prefix of those attributes. In the case of the histological diagnosis, an additional column indicating missing values was added. Lastly, the column `diagnosis_melanoma` can be interpreted also as a binary diagnosis in the form 1=malignant, 0=benign.
