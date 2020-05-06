"""
Tests CatBoost CatBoostRegressor converter.
"""
import unittest
import numpy
import catboost
import sys

from catboost.utils import convert_to_onnx_object
from sklearn.datasets import make_regression
from onnxmltools.convert import convert_catboost
from onnxmltools.utils import dump_data_and_model, dump_single_regression, dump_multiple_regression


class TestCatBoostRegressor(unittest.TestCase):

    def test_catboost_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)

        catboost_model = catboost.CatBoostRegressor(task_type='CPU', loss_function='RMSE', n_estimators=100, verbose=0)
        dump_single_regression(catboost_model)

        catboost_model.fit(X, y)
        catboost_onnx = convert_to_onnx_object(catboost_model)

        self.assertTrue(catboost_onnx is not None)
        # fails here
        dump_data_and_model(X.astype(numpy.float32), catboost_model, catboost_onnx, basename="CatBoostRegressor")

if __name__ == "__main__":
    unittest.main()
