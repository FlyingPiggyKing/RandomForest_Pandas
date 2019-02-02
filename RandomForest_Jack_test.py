import unittest
import pandas as pd
from  RandomForest_Jack import cal_gini
import numpy as np
# from random_forest import gini_index
from  RandomForest_Jack import get_split
from RandomForest_Jack import to_leaf

class RandomForest_Jack_test(unittest.TestCase):
    def test_cal_gini(self):
        df_left = pd.DataFrame([[1, 0], [2, 0], [3, 1]], columns=['attr', 'res'])
        df_right = pd.DataFrame([[10, 1], [2, 0], [5, 1]], columns=['attr', 'res'])
        gini  =  cal_gini([df_left, df_right], [0, 1])
        gini_str = '%.2f' % gini
        expect_value = 1 - np.square(2 / 3) - np.square(1 / 3)
        expect_str = '%.2f' % expect_value
        self.assertEqual(gini_str, expect_str)
        # gini2 = gini_index([[[1, 0], [2, 0], [3, 1]],[[10, 1], [2, 0], [5, 1]]], [0, 1])
        # gini2_str = '%.2f' % gini2
        # self.assertEqual(gini2_str, expect_str)

    def test_get_split(self):
        df = pd.DataFrame([[1, 2, 0], [3, 4, 1], [5, 6, 0]],
                          columns=['col1', 'col2', 'result'])
        print('The test data groups are as below:')
        print(df.head())
        expected_groups = \
            pd.DataFrame([['None', '0, 1, 0'], ['0', '1, 0'], ['0, 1', '0'],
                                      ['None', '0, 1, 0'], ['0', '1, 0'], ['0, 1', '0']],
                         columns=['left_group', 'right_group'],
                         index=['gini_split0', 'gini_split1', 'gini_split2',
                                'gini_split3', 'gini_split4', 'gini_split5'])

        print('The expect gini group split are as below:')
        print(expected_groups.head())
        gini0 = float(1) - np.square(float(2) / 3) - np.square(float(1) / 3)
        gini1 = (float(1) - np.square(float(1) / 2) - \
                 np.square(float(1) / 2)) * (float(2) / 3)
        gini2 = (float(1) - np.square(float(1) / 2) - \
                 np.square(float(1) / 2)) * (float(2) / 3)
        expected_gini = min(gini0, gini1, gini2)
        print('The expect minimal gini: %.5f' % expected_gini)
        actual_gini = get_split(df, 2)['gini']
        print('actual calculated gini value is %.5f' % actual_gini)
        self.assertEqual(expected_gini, actual_gini)

    def test_to_leaf(self):
        df = pd.DataFrame([[1, 2, 0], [3, 4, 1], [5, 6, 0]],
                          columns=['col1', 'col2', 'result'])
        expected_leaf = 0
        actual_leaf = to_leaf(df)
        self.assertEqual(expected_leaf, actual_leaf)
