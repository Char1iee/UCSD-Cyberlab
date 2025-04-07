import numpy as np
import pandas as pd
from sklearn import preprocessing


def _add_one_hot_encoding(df: pd.DataFrame, metadata_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
        return updated df and metadata_df, after one-hot-encoding all categorical features in df
    """

    # list the features to one-hot-encode
    cat_features = metadata_df[metadata_df.type == 'categorical'].feature_name.to_list()

    # TODO [OPTIONAL] consider avoiding one-hot encoding binary features
    # cats_to_one_hot = []
    # for cat_feature in cat_features:
    #     if df[cat_feature].nunique() > 2:
    #         cats_to_one_hot.append(cat_feature)

    # one-hot-columns will be of form "{original_categorical_feature}==={encoded_value}"
    df = pd.get_dummies(df, columns=cat_features, prefix_sep="===", drop_first=False, dtype=np.float32)
    # move label column to the end
    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()
    df = df[[col for col in df.columns if col != label_col] + [label_col]]

    # updated metadata df, with new one-hot coordinates
    new_metadata_rows = []

    for feature_name in df.columns:
        # add the row to the new_metadata_df, in the simplest way
        # get the dict of the row where 'metadata_df.feature_name == feature_name'
        if feature_name.split('===')[0] in cat_features:
            # if the feature was one-hot encoded, add a row for its one-hot-encoding coordinate
            feature_name, encoded_val = feature_name.split('===')  # split to original feature name and encoded val
            new_metadata_row = metadata_df[metadata_df.feature_name == feature_name].to_dict(orient='records')[0]
            new_metadata_row.update({'range': '[0, 1]', 'one_hot_encoding': encoded_val})
        else:  # if the feature was not one-hot encoded, add it as is
            new_metadata_row = metadata_df[metadata_df.feature_name == feature_name].to_dict(orient='records')[0]

        new_metadata_rows.append(new_metadata_row)

    return df, pd.DataFrame(new_metadata_rows)


def _add_mapping_encoding(df: pd.DataFrame, metadata_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
        return updated df and metadata_df, after mapping all categorical features in df to integers
    """
    # list the features to map
    metadata_df['encoding_map'] = None
    for cat_feature in metadata_df[metadata_df.type == 'categorical'].feature_name:
        # encode categories in integers
        le = preprocessing.LabelEncoder()
        df[cat_feature] = le.fit_transform(df[cat_feature])
        metadata_df.loc[metadata_df.feature_name == cat_feature, 'encoding_map'] = [{idx: actual_val for idx, actual_val in
                                                                             enumerate(le.classes_.tolist())}]
        metadata_df.loc[metadata_df.feature_name == cat_feature, 'range'] = f'[0, {len(le.classes_) - 1}]'
    return df, metadata_df


def add_categorical_encoding(df: pd.DataFrame, metadata_df: pd.DataFrame, encoding_method='one_hot_encoding') -> \
        (pd.DataFrame, pd.DataFrame):
    if encoding_method == 'one_hot_encoding':
        # count number of unique values in each categorical feature
        df, metadata_df = _add_one_hot_encoding(df, metadata_df)
    elif encoding_method is None:
        # Default encoding follows a simple mapping of categories to integers
        df, metadata_df = _add_mapping_encoding(df, metadata_df)
    else:
        raise ValueError(f'encoding_method={encoding_method} is not supported')

    return df, metadata_df
