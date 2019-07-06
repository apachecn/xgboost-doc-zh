# 贡献指南

> 请您勇敢地去翻译和改进翻译。虽然我们追求卓越，但我们并不要求您做到十全十美，因此请不要担心因为翻译上犯错——在大部分情况下，我们的服务器已经记录所有的翻译，因此您不必担心会因为您的失误遭到无法挽回的破坏。（改编自维基百科）

可能有用的链接：

+ [0.72 英文文档](https://xgboost.readthedocs.io/en/release_0.72)

负责人：

* [那伊抹微笑](https://github.com/wangyangting): 1042658081
* [Peppa](https://github.com/chenyyx): 190442212

## 章节列表

+   [Introduction](README.md)
+   [开始使用 XGBoost](docs/1.md)
+   [XGBoost 教程](docs/2.md)
    +   [Boosted Trees 介绍](docs/3.md)
    +   [AWS 上的分布式 XGBoost YARN](docs/4.md)
    +   [DART booster](docs/5.md)
+   [XGBoost 入门指引](docs/6.md)
    +   安装
        +   [安装指南](docs/7.md)
    +   以特定的方式使用 XGBoost
        +   [参数调整注意事项](docs/8.md)
        +   [使用 XGBoost 外部存储器版本（测试版）](docs/9.md)
    +   开发和破解 XGBoost
        +   [为 XGBoost 做贡献](docs/10.md)
    +   [常见问题](docs/11.md)
+   [XGBoost Python Package](docs/12.md)
    +   [Python 软件包介绍](docs/13.md)
    +   [Python API 参考](docs/14.md)
+   [XGBoost 参数](docs/15.md)

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
+   将译文放在`docs/0.72`文件夹下
+   `push`
+   `pull request`

请见 [Github 入门指南](https://github.com/apachecn/kaggle/blob/master/docs/GitHub)。
