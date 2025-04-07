import pandas as pd
from typing import Tuple, List

from scipy.io import arff

from src.datasets.preprocess.utils import add_categorical_encoding


def get_phishing_dataset(data_file_path: str,
                         metadata_file_path: str,
                         encoding_method: str = None) -> Tuple[pd.DataFrame, List[str], str]:
    # 0. Read data
    data = arff.loadarff(data_file_path)
    df = pd.DataFrame(data[0])
    metadata_df = pd.read_csv(metadata_file_path)

    # 1. Set order and filtering of columns from metadata (from metadata file)
    df = df[metadata_df['feature_name']].copy()

    # 2. Perform basic transformations (missing data, mapping columns, etc)
    for cat_feature in metadata_df[metadata_df.type == 'categorical'].feature_name:
        # Missing: Missing val --> most common val
        missing_val_string = ' ?'
        most_common_val = df[cat_feature].mode()[0]
        df[cat_feature].replace(missing_val_string, most_common_val, inplace=True)

    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()  # label_col = 'CLASS_LABEL'
    df[label_col] = df[label_col] == b'1'

    # 3. Categorical encoding:
    df, metadata_df = add_categorical_encoding(df, metadata_df, encoding_method=encoding_method)

    # 5. split to input and labels
    x_df, y_df = df.drop(columns=[label_col]), df[label_col]

    return x_df, y_df, metadata_df
