import pandas as pd
import numpy as np

def try_split(column, value, df):
    left_df = df[df[column] < value]
    right_df = df[df[column] >= value]
    return left_df, right_df

def cal_gini(dfs, classes):
    parent_len = float(sum([len(df.index) for df in dfs]))

    gini = 0.0
    for df in dfs:
        df_len = float(len(df.index))
        if df_len == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = len(df[df.iloc[:, -1] == class_val].index) / df_len
            score += p * p

        gini += (1 - score) * (df_len / parent_len)
    return gini

def get_split(df, num_feature):
    classes = df.iloc[:, -1].value_counts().index.tolist()
    b_column, b_value, b_score, b_groups = None, None, float(1.0), None
    all_features = df.columns.values[:-1]
    #sample the features
    features = np.random.choice(all_features,
                                num_feature, replace=False)
    for fea in features:
        for row_i, row in df.iterrows():
            df_groups = try_split(fea, row[fea], df)
            gini = cal_gini(df_groups, classes)
            if gini < b_score:
                b_column, b_value, b_score, b_groups = fea, row[fea], \
                                                       gini, df_groups
    return {'index':b_column, 'value':b_value,
            'groups':b_groups, 'gini': b_score}


def to_leaf(df):
    return df.iloc[:, -1].value_counts().index.values[0]