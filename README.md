# ClassSR_paddle
ClassSR_paddle
## 一、简介
本项目采用百度飞桨框架paddlepaddle复现：ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic, by Jiaqing Zhang and Kai jiang (张佳青&蒋恺)


paper：[ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)
code：[ClassSR](https://github.com/Xiangtaokong/ClassSR)

本代码包含了原论文的默认配置下的训练和测试代码。

## 二、复现结果

![Generated Results]()

## 三、环境依赖

```
python -m pip install -r requirements.txt
```

此代码在python 3.7中进行了测试

## 四、实现

### 训练

#### 准备数据集

通过百度云链接下载数据集：[DIV2K](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg) 提取码：jcty
按顺序运行以下代码处理数据集
注意修改路径（只需要修改/home/aistudio/DIV2K目录，后续子文件名字可以不用修改）

```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py (用于训练SR分支，可不运行)
```

#### 训练SR_MODEL(RCAN) (可不运行)

- 需要在[train_RCAN.yml](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg)修改数据集路径（只需要修改/home/aistudio/DIV2K目录，后续子文件名字可以不用修改）
```
python train.py -opt options/train/train_RCAN.yml
```
已将原作者的权重进行转换，进行了测试对比是一致的，并且进行了从零训练的对比，第一个batch_size出来的结果是一致的
权重下载路径[DIV2K](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd
有三个分支的权重，分别是RCAN_branch1.pdparams、RCAN_branch2.pdparams、RCAN_branch3.pdparams

#### 训练Class_MODEL(Class_RCAN)
- 需要在[train_ClassSR_RCAN.yml](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg)修改数据集路径（只需要修改/home/aistudio/DIV2K目录，后续子文件名字可以不用修改），修改三个分支权重的路径（只需要修改/home/aistudio/model_pretrained目录，后续子文件名字可以不用修改）
```
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```

### 测试

- 需要在[test_ClassSR_RCAN.yml](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg)修改数据集路径（只需要修改/home/aistudio/DIV2K目录，后续子文件名字可以不用修改）
- 权重下载路径[DIV2K](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd 下载权重ClassSR_RCAN.pdparams。需要在[test_ClassSR_RCAN.yml](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg)修改权重路径
```
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```

## 五、代码结构


```
├──   # 存放模型文件的路径
├── Input  # 存放数据集的路径
├── Output  # 存放程序输出的路径
    ├── log.txt #日志文件
├──   # 定义模型，工具等
├── test.py  # 评估程序
├── README.md
├── train.py  # 训练程序
├── config.py #定义一些参数
├── requirements.txt #所需环境

```

## 六、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张佳青 |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像超分 |
| 下载链接 | [预训练模型]() 提取码： |
| 飞桨项目 | [欢迎fork]() |
|  数据集  | [DIV2K]() |
