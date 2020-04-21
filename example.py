import onnxmltools
import onnx
import catboost
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=123)

model = catboost.CatBoostRegressor(task_type='CPU', loss_function='RMSE')

model.fit(X_train, y_train, verbose=100, eval_set=(X_test, y_test), )

### Saving Model

# model.save_model(
#     "boston.onnx",
#     format="onnx",
#     export_parameters={
#         'onnx_domain': 'ai.catboost',
#         'onnx_model_version': 1,
#         'onnx_doc_string': 'test model for boston dataset',
#         'onnx_graph_name': 'CatBoostModel_for_BinaryClassification'
#     }
# )

onnx_model = onnxmltools.convert_catboost(model, name='catboost_example')

onnxmltools.utils.save_model(onnx_model, 'catboost.onnx')

model_onnx = onnx.load("catboost.onnx")
onnx.checker.check_model(model_onnx)
