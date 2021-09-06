# ClassSR_paddle
## 一、简介
本项目采用百度飞桨框架paddlepaddle复现：ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic, by Jiaqing Zhang and Kai jiang (张佳青&蒋恺)


paper：[ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)

code：[ClassSR](https://github.com/Xiangtaokong/ClassSR)

本代码包含了原论文的默认配置下的训练和测试代码。

## 二、复现结果

| - | Model | Test2K | FLOPs |
|  ----  |  ----  |  ----  |  ----  |
| 原论文 | ClassSR-RCAN | 26.39dB | 21.22G(65%) |
| 复现 | ClassSR-RCAN | 26.39dB | 23.06(70.73%) |

![Results](https://github.com/icey-zhang/ClassSR_paddle/blob/main/results/ClassSR_result.png)

## 三、环境依赖

```
python -m pip install -r requirements.txt
```

此代码在python 3.7中进行了测试

## 四、实现

### 训练 - 训练Class_MODEL(Class_RCAN)

#### 下载数据集

通过百度云链接下载数据集：

有两种方式

- 下载原数据集再用代码处理
- 直接下载处理好的数据集(建议)

  1. 下载原数据集再用代码处理
      
      [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104667)

      DIV2K_train_HR.zip(训练集)

      DIV2K_train_LR_bicubic_X4.zip(训练集)

      val_10.zip(验证集)

      按顺序运行以下代码处理数据集

      **注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）

      ```
      cd codes/data_scripts
      python data_augmentation.py
      python generate_mod_LR_bic.py
      python extract_subimages_train.py
      ```

  2. 直接下载处理好的数据集
  
      [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748)

      DIV2K_scale_sub.zip(训练集)

      val_10.zip(验证集)
      

#### 下载SR_MODEL(RCAN)的权重

有三个分支的权重，分别是RCAN_branch1.pdparams、RCAN_branch2.pdparams、RCAN_branch3.pdparams。[model_pretrained](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd

#### 修改路径

需要在[train_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/train/train_ClassSR_RCAN.yml)修改数据集路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改），修改三个分支权重的路径（只需要修改/home/aistudio/model_pretrained目录，后续子文件名字可以不用修改）

#### 开始训练

```
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```

### 测试
#### 下载数据集

通过百度云链接下载数据集：

有两种方式

- 下载原数据集再用代码处理
- 直接下载处理好的数据集 (建议)

  1. 下载原数据集再用代码处理
  
      [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/55117)
      
      trainHR_1201to1400.zip(测试集)
      
      按顺序运行以下代码处理数据集

      **注意修改路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字可以不用修改）

      ```
      cd codes/data_scripts
      python test2k.py
      ```

  2. 直接下载处理好的数据集
  
     [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748)

      test2K.zip(测试集)

#### 下载Class_MODEL(Class_RCAN)的权重

下载权重latest_G.pdparams。[model_pretrained](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd。

#### 修改路径

需要在[test_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/test/test_ClassSR_RCAN.yml)修改数据集路径（只需要修改/home/aistudio/data_div2k目录，后续子文件名字不用修改）。修改权重路径（只需修改/home/aistudio/model_pretrained目录，后续子文件夹名字不用修改）

#### 开始测试

```
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```

## 五、代码结构


```
./ClassSR_paddle
├─data             
├─data_scripts                                          
├─models               #模型
├─options              #配置文件
├─results              #日志文件
├─utils                #一下API                                               
|  README.md                               
│  train.py            #分支训练
│  test.py             #分支测试
│  train_ClassSR.py    #ClassSR训练
│  test_ClassSR.py     #ClassSR测试

```

## 六、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张佳青 |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像超分 |
| 模型权重 | [model_pretrained](https://pan.baidu.com/s/1B4DdsBDaiH74uwcp-oMosw) 提取码：zxpd |
| 飞桨项目 | [欢迎fork](https://aistudio.baidu.com/aistudio/projectdetail/2313539?shared=1) |
|  数据集  | [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104667) [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748) [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/55117) |
| 测试结果 | [test2K](https://pan.baidu.com/s/1SBZqFHAy3FG-RZzBfNnefg) 提取码：dan9 |
