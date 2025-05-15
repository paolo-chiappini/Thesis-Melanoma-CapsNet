import pandas as pd
from os import path

def preprocess_metadata(root, raw_csv, output_csv):
    df = pd.read_excel(path.join(root, raw_csv))

    df.columns = df.columns.str.replace('\n', ' ', regex=True).str.strip()
    colname_mapping = {
        "Image Name": "image_name",
        "Histological Diagnosis": "histological_diagnosis",
        "Common Nevus": "diagnosis_common_nevus",
        "Atypical Nevus": "diagnosis_atypical_nevus",
        "Melanoma": "diagnosis_melanoma",
        "Asymmetry (0/1/2)": "asymmetry",
        "Pigment Network (AT/T)": "pigment_network",
        "Dots/Globules (A/AT/T)": "dots_globules",
        "Streaks (A/P)": "streaks",
        "Regression Areas (A/P)": "regression_areas",
        "Blue-Whitish Veil (A/P)": "blue_whitish_veil",
        "White": "color_white",
        "Red": "color_red",
        "Light-Brown": "color_light_brown",
        "Dark-Brown": "color_dark_brown",
        "Blue-Gray": "color_blue_gray_brown",
        "Black": "color_black",
    }
    df.rename(columns=colname_mapping, inplace=True)

    assert 'histological_diagnosis' in df.columns, 'Error during preprocessing: could not find histological_diagnosis'

    cols_to_fill = df.columns[df.columns.get_loc('histological_diagnosis') + 1:]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    df[cols_to_fill] = df[cols_to_fill].replace({
        'X': 1, 'A': 0, 'P': 1, 'AT': 0, 'T': 1
    })

    assert 'asymmetry' in df.columns, 'Error during preprocessing: could not find asymmetry'

    df['asymmetry'] = df['asymmetry'].map({
        0: 'fully_symmetric',
        1: 'symmetric_1_axis',
        2: 'asymmetric'
    })
    
    
    df_hist_diag_encoded = pd.get_dummies(df['histological_diagnosis'], dummy_na=True, drop_first=False, dtype=int).add_prefix('hist_')
    df_asymm_encoded = pd.get_dummies(df['asymmetry'], dummy_na=False, drop_first=False, dtype=int).add_prefix('asymmetry_')

    df_new = pd.concat([df['image_name'], df_hist_diag_encoded, df_asymm_encoded, df.loc[:, ~df.columns.str.contains('|'.join(['image_name', 'asymmetry', 'hist']))]], axis=1)
    df_new.columns = df_new.columns.str.replace(' ', '_', regex=True).str.strip().str.lower()
    
    df_new.to_csv(path.join(root, output_csv), index=False)
        
if __name__ == '__main__':
    import argparse
    pd.set_option('future.no_silent_downcasting', True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    preprocess_metadata(args.root, args.raw, args.out)
    print(f'> Finished preprocessing dataset: results available at {path.join(args.root, args.out)}')
    
    
    