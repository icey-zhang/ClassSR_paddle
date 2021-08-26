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

#### 训练SR_MODEL(RCAN) (可不运行，不用管)
通过百度云链接下载数据集：[DIV2K](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg) 提取码：jcty
有两种方式
- 下载原数据集再用代码处理
- 直接下载处理好的数据集再分成子类

1. 下载原数据集再用代码处理
DIV2K_train_HR.zip(训练集)
DIV2K_train_LR_bicubic_X4.zip(训练集)
Set5.zip(验证集)
按顺序运行以下代码处理数据集
注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）
```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py
```

2. 直接下载处理好的数据集再分成子类
DIV2K_scale_sub.zip(训练集)
Set5.zip(验证集)
按顺序运行以下代码处理数据集
注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）
```
cd codes/data_scripts
python divide_subimages_train.py
```

- 需要在[train_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/train/train_RCAN.yml)修改数据集路径（只需要修改/home/aistudio/DIV2K目录，后续子文件名字不用修改）
```
python train.py -opt options/train/train_RCAN.yml
```
已将原作者的权重进行转换，进行了测试对比是一致的，并且进行了从零训练的对比，第一个batch_size出来的结果是一致的
权重下载路径[model_pretrained](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd
有三个分支的权重，分别是RCAN_branch1.pdparams、RCAN_branch2.pdparams、RCAN_branch3.pdparams

#### 训练Class_MODEL(Class_RCAN)
##### 下载数据集
通过百度云链接下载数据集：[DIV2K](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg) 提取码：jcty
有两种方式
- 下载原数据集再用代码处理
- 直接下载处理好的数据集

1. 下载原数据集再用代码处理
DIV2K_train_HR.zip(训练集)
DIV2K_train_LR_bicubic_X4.zip(训练集)
val_10.zip(验证集)
按顺序运行以下代码处理数据集
注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）

```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
```
2. 直接下载处理好的数据集
DIV2K_scale_sub.zip(训练集)
val_10.zip(验证集)
- 需要在[train_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/train/train_RCAN.yml)修改数据集路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改），修改三个分支权重的路径（只需要修改/home/aistudio/model_pretrained目录，后续子文件名字可以不用修改）
```
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```

### 测试
通过百度云链接下载数据集：[DIV2K](https://pan.baidu.com/s/12eTTMe_yk7WgQ7aZnYbnDg) 提取码：jcty
有两种方式
- 下载原数据集再用代码处理
- 直接下载处理好的数据集

1. 下载原数据集再用代码处理
DIV2K_valid_HR.zip(测试集)
DIV2K_valid_LR_bicubic_X4.zip(测试集)
按顺序运行以下代码处理数据集
注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字可以不用修改）
```
cd codes/data_scripts
python extract_subimages_test.py
```
2. 直接下载处理好的数据集
DIV2K_valid_HR_sub.zip(测试集)

- 需要在[test_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/test/test_ClassSR_RCAN.yml)修改数据集路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）
- 权重下载路径[model_pretrained](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd 下载权重ClassSR_RCAN.pdparams。需要在[test_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/test/test_ClassSR_RCAN.yml)修改权重路径（只需修改/home/aistudio目录，后续子文件夹名字不用修改）
```
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```

## 五、代码结构


```
├──   # 之后再放

```

## 六、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张佳青 |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像超分 |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd |
| 飞桨项目 | [欢迎fork]() |
|  数据集  | [DIV2K]() |
