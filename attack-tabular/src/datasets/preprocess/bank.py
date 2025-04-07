import pandas as pd
from typing import Tuple, List

from src.datasets.preprocess.utils import add_categorical_encoding

IDX_TO_MONTH = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def get_bank_dataset(data_file_path: str,
                     metadata_file_path: str,
                     encoding_method: str = None) -> Tuple[pd.DataFrame, List[str], str]:

    # 0. Read data
    df = pd.read_csv(data_file_path, sep=";")
    metadata_df = pd.read_csv(metadata_file_path)

    # 1. Set order and filtering of columns from metadata (from metadata file)
    df = df[metadata_df['feature_name']].copy()

    # 2. Perform basic transformations (missing data, mapping columns, etc)
    # 2.1. Ordinal mapping

    # `month` feature [DISABLED]
    # month_mapping = {month: (idx + 1) for idx, month in enumerate(IDX_TO_MONTH)}
    # df['month'] = df['month'].map(month_mapping)
    # metadata_df.loc[metadata_df.feature_name == 'month', 'type'] = 'ordinal'
    # metadata_df.loc[metadata_df.feature_name == 'month', 'range'] = '[1,12]'

    # `education` feature
    education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    df['education'] = df['education'].map(education_mapping)
    metadata_df.loc[metadata_df.feature_name == 'month', 'type'] = 'ordinal'
    metadata_df.loc[metadata_df.feature_name == 'month', 'range'] = '[0,3]'

    # `pdays` feature
    # pdays means how many days past after last contact.
    #   So we should change -1(non-called) values to something big like 999.
    df.loc[(df.pdays == -1), "pdays"] = 999

    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()  # label_col = 'y'
    df[label_col] = (df[label_col] == 'yes')

    # 3. Categorical encoding:
    df, metadata_df = add_categorical_encoding(df, metadata_df, encoding_method=encoding_method)

    # 5. split to input and labels
    x_df, y_df = df.drop(columns=[label_col]), df[label_col]

    return x_df, y_df, metadata_df

