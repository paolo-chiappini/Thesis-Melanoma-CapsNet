import pandas as pd


def summarize_evaluation(recon_results, mi_results, attribute_names=None):
    df_mi = pd.DataFrame(mi_results, columns=attribute_names)
    df_recon = pd.DataFrame([recon_results])
    return df_recon, df_mi
