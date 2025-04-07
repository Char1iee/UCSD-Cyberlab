import pandas as pd
from typing import Tuple, List

from src.datasets.preprocess.utils import add_categorical_encoding


def get_adult_dataset(data_file_path: str,
                      metadata_file_path: str,
                      encoding_method: str = None) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param data_file_path: path to raw CSV dataset of the bank.
    :param metadata_file_path: path to raw CSV dataset of the bank.
        # metadata files include information essential for preprocessing, structure constraints and for inference time.
        # The metadata corresponding dataframe is also updated after the preprocessing.
    :param encoding_method: whether to perform one-hot-encoding on categorical features in the dataset
    :returns: DataFrame after pre-processing (including encoding categorical features),
                and a list of the features-meta-data
    """

    # 0. Read data
    raw_data_header = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status',
                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(data_file_path, names=raw_data_header)
    metadata_df = pd.read_csv(metadata_file_path)

    # 1. Set order and filtering of columns from metadata (from metadata file)
    df = df[metadata_df['feature_name']].copy()

    # 2. Perform basic transformations (missing data, mapping columns, etc)
    for cat_feature in metadata_df[metadata_df.type == 'categorical'].feature_name:
        # Missing: Missing val --> most common val
        missing_val_string = ' ?'
        most_common_val = df[cat_feature].mode()[0]
        df[cat_feature].replace(missing_val_string, most_common_val, inplace=True)

    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()
    df[label_col] = (df[label_col] == ' >50K')  # we predict who _has_ high income

    # 3. Categorical encoding:
    df, metadata_df = add_categorical_encoding(df, metadata_df, encoding_method=encoding_method)

    # 5. split to input and labels
    x_df, y_df = df.drop(columns=[label_col]), df[label_col]

    return x_df, y_df, metadata_df

