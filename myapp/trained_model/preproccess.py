import pandas as pd

def dummy_encode(df, column):
    dummies = pd.get_dummies(df[column], prefix=column, prefix_sep='_', drop_first=True)
    dummies = dummies.map(lambda x: 1 if x == 1 else -1)
    return pd.concat([df.drop(column, axis=1), dummies], axis=1)


def preprocess_input(input_df, min_vals, max_vals, categorical_columns, expected_features):
    """
    Preprocess a new input for model prediction.

    Args:
        input_df (pd.DataFrame): A DataFrame with one or more rows to predict.
        min_vals (pd.Series): Minimum values used in training (index must match numeric cols).
        max_vals (pd.Series): Maximum values used in training.
        categorical_columns (list): List of categorical columns.
        expected_features (list): Final feature names the model expects in order.

    Returns:
        pd.DataFrame: Preprocessed input ready for the model.
    """

    # Normalize numeric columns using -1 to 1 scaling
    numeric_cols = [col for col in input_df.columns if col not in categorical_columns]
    input_df[numeric_cols] = -1 + 2 * ((input_df[numeric_cols] - min_vals[numeric_cols]) / (max_vals[numeric_cols] - min_vals[numeric_cols]))

    # Encode categorical columns
    for cat_col in categorical_columns:
        input_df = dummy_encode(input_df, cat_col)

    # Add missing dummy columns with default -1 (as would be in training)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = -1

    # Reorder columns to match model input
    input_df = input_df[expected_features]

    return input_df

