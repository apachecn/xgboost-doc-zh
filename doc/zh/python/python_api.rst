Python API 参考
====================
该页面提供了有关 xgboost 的 Python API 参考, 请参阅 Python 软件包介绍以了解更多关于 python 软件包的信息.

该页面中的文档是由 sphinx 自动生成的. 其中的内容不会在 github 上展示出来, 你可以在 http://xgboost.apachecn.org/cn/latest/python/python_api.html 页面上浏览它.

核心的数据结构
-------------------
.. automodule:: xgboost.core

.. autoclass:: xgboost.DMatrix
    :members:
    :show-inheritance:

.. autoclass:: xgboost.Booster
    :members:
    :show-inheritance:


学习的 API
------------
.. automodule:: xgboost.training

.. autofunction:: xgboost.train

.. autofunction:: xgboost.cv


Scikit-Learn 的 API
----------------
.. automodule:: xgboost.sklearn
.. autoclass:: xgboost.XGBRegressor
    :members:
    :show-inheritance:
.. autoclass:: xgboost.XGBClassifier
    :members:
    :show-inheritance:

绘图的 API
------------
.. automodule:: xgboost.plotting

.. autofunction:: xgboost.plot_importance

.. autofunction:: xgboost.plot_tree

.. autofunction:: xgboost.to_graphviz
