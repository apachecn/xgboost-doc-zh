# 贡献指南

> 请您勇敢地去翻译和改进翻译。虽然我们追求卓越，但我们并不要求您做到十全十美，因此请不要担心因为翻译上犯错——在大部分情况下，我们的服务器已经记录所有的翻译，因此您不必担心会因为您的失误遭到无法挽回的破坏。（改编自维基百科）

可能有用的链接：

+ [0.72 中文文档](https://xgboost.apachecn.org/docs/0.72)
+ [0.72 英文文档](https://xgboost.readthedocs.io/en/release_0.72)

负责人：

* [1266](https://github.com/wangweitong): 1097828409
* [腻味](https://github.com/xxxx): 1185685810

## 章节列表

* [Introduction](README.md)
* [安装指南](https://xgboost.readthedocs.io/en/latest/build.html)
* [XGBoost入门](https://xgboost.readthedocs.io/en/latest/get_started.html)
* [XGBoost教程](https://xgboost.readthedocs.io/en/latest/tutorials/index.html)
    * [Boosted Trees简介](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
    * [使用AWS YARN分布式XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/aws_yarn.html)
    * [使用XGBoost4J-Spark的分布式XGBoost](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html)
    * [DART助推器](https://xgboost.readthedocs.io/en/latest/tutorials/dart.html)
    * [单调约束](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)
    * [XGBoost中的随机森林](https://xgboost.readthedocs.io/en/latest/tutorials/rf.html)
    * [特征交互约束](https://xgboost.readthedocs.io/en/latest/tutorials/feature_interaction_constraint.html)
    * [DMatrix的文本输入格式](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    * [参数调整注意事项](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
    * [使用XGBoost外部存储器版本（测试版）](https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html)
    * [自定义目标和评估指标](https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html)
* [经常问的问题](https://xgboost.readthedocs.io/en/latest/faq.html)
* [XGBoost用户论坛](https://discuss.xgboost.ai/)
* [GPU支持](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
* [XGBoost参数](https://xgboost.readthedocs.io/en/latest/parameter.html)
* [Python包](https://xgboost.readthedocs.io/en/latest/python/index.html)
    * [Python包介绍](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
    * [Python API参考](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
    * [Python示例](https://github.com/dmlc/xgboost/tree/master/demo/guide-python)
* [R包](https://xgboost.readthedocs.io/en/latest/R-package/index.html)
    * [R中的XGBoost简介](https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html)
    * [使用XGBoost了解您的数据集](https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html)
* [JVM包](https://xgboost.readthedocs.io/en/latest/jvm/index.html)
    * [XGBoost4J入门](https://xgboost.readthedocs.io/en/latest/jvm/java_intro.html)
    * [XGBoost4J-Spark教程](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html)
    * [代码示例](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example)
    * [XGBoost4J Java API](https://xgboost.readthedocs.io/en/latest/jvm/javadocs/index.html)
    * [XGBoost4J Scala API](https://xgboost.readthedocs.io/en/latest/jvm/scaladocs/xgboost4j/index.html)
    * [XGBoost4J-Spark Scala API](https://xgboost.readthedocs.io/en/latest/jvm/scaladocs/xgboost4j-spark/index.html)
    * [XGBoost4J-Flink Scala API](https://xgboost.readthedocs.io/en/latest/jvm/scaladocs/xgboost4j-flink/index.html)
* [Julia包](https://xgboost.readthedocs.io/en/latest/julia.html)
* [CLI界面](https://xgboost.readthedocs.io/en/latest/cli.html)
* [有助于XGBoost](https://xgboost.readthedocs.io/en/latest/contribute.html)

## 流程

### 一、认领

首先查看[整体进度](https://github.com/apachecn/pytorch-doc-zh/issues/274)，确认没有人认领了你想认领的章节。
 
然后回复 ISSUE，注明“章节 + QQ 号”（一定要留 QQ）。

### 二、翻译

可以合理利用翻译引擎（例如[谷歌](https://translate.google.cn/)），但一定要把它变得可读！

可以参照之前版本的中文文档，如果有用的话。

如果遇到格式问题，请随手把它改正。

### 三、提交

**提交的时候不要改动文件名称，即使它跟章节标题不一样也不要改，因为文件名和原文的链接是对应的！！！**

+   `fork` Github 项目
+   将译文放在`docs/0.90`文件夹下
+   `push`
+   `pull request`

请见 [Github 入门指南](https://github.com/apachecn/kaggle/blob/master/docs/GitHub)。
