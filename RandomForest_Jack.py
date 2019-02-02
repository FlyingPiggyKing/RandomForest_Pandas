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

def split(node, max_depth, min_size, num_feature, depth):
    left_df, right_df = node['groups']
    del(node['groups'])
    #if only left or only right, it means we already choose
    #the attribute and value to split the df and get a minium
    #gini, if we continue to split the df, we will get the same
    #result, which is meaningless, so we output the leaf
    if len(left_df.index) == 0 or len(right_df.index) == 0:
        node['left'], node['right'] = \
            to_leaf(pd.concat(left_df, right_df))
        return
    #check for max depth
    if depth > max_depth:
        node['left'], node['right'] = \
            to_leaf(left_df), to_leaf(right_df)
        return
    #process left child
    if len(left_df.index) <= min_size:
        node['left'] = to_leaf(left_df)
    else:
        node['left'] = get_split(left_df, num_feature)
        split(node['left'], max_depth, min_size, num_feature, depth+1)
    #process the right node
    if len(right_df.index) <= min_size:
        node['right'] = to_leaf(right_df)
    else:
        node['right'] = get_split(right_df, num_feature)
        split(node['right'], max_depth, min_size,num_feature, depth + 1)

# Build a decision tree
def build_tree(data_df, max_depth, num_feature, min_size):
    root = get_split(data_df, num_feature)
    split(root, max_depth, min_size, num_feature, 1)
    return root

#make prediction function
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            predict(node['right'], row)
        else:
            return node['right']

def voting_predict(trees, row):
    predicts = [predict(tree, row) for tree in trees]
    final_predict = max(set(predicts), predicts.count())
    return final_predict

def random_forest(data_df, num_trees, max_depth,
                  min_size, num_sample, num_feature):
    trees = []
    for tree_i in range(num_trees):
        sample_df = data_df.sample(num_sample, replace=True)
        tree = build_tree(sample_df, max_depth, num_feature, min_size)
        trees.append(tree)
    return trees