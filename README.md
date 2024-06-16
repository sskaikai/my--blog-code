# 渔船作业方式识别
## 1.课设选题来源和内容
### 选题来源：DataFountain - 数据科学竞赛创新平台
### 选题内容：
1.1赛题名称：渔船作业方式识别 
1.2赛题背景：随着大数据发展和轨迹数据挖掘的深入研究，渔船轨迹数据为渔业数据挖掘及相关应用提供非常好的机会。海洋捕捞渔船作业类型分为多种,主要有拖网、围网、张网、刺网等方式,违规捕捞作业将会对海洋生态环境和渔业资源产生严重的影响，同时也给渔业管理带来困难。因此，需要依据真实渔船轨迹数据（已脱敏）利用机器学习相关技术，建立稳健的渔船作业方式识别模型，用于准确识别渔船的作业方式，便于渔业管理的同时也为后续渔业科技研究带来便利
1.3赛题任务：依据真实渔船轨迹数据（经纬度、速度、方向、时间等），利用机器学习相关技术，建立稳健的渔船作业方式识别模型，有效识别渔船的作业方式。
## 2.数据集描述
数据整理自东海船舶的轨迹数据（已脱敏），为海上真实的船舶历史轨迹数据，数据集涵盖多个维度的信息，每一条轨迹数据包括渔船ID、经纬度、速度、方向、时间、作业方式（拖网、围网和刺网）等信息。
已给数据包括一个训练集train_data，一个测试集test_data，一个提交样例submission.csv
## 3.评测方案
需要以csv文件格式提交，提交模型结果到大数据竞赛平台，平台进行在线评分，实时排名。
## 4.课设设计说明（算法与模型）  
在实验初阶段，我遇到了一系列问题，类似的训练集数据解压失败，训练集与测试集以文件形式存储，对于某一个渔船编号不仅仅只有一条训练数据，而是拥有许多不同的训练特征，比如18330号渔船，其不同的经纬度，时间对应不同的作业方式（type)，还有特征提取失败，时间序列无法读取，数据出现乱码等等这样的问题对于数据预处理，模型选取与特征提取都是很大的挑战。在尝试多次以报错告终后，我将训练集中的大部分的渔船编号对应数据存储在一个csv里面，对其进行数据预处理，并进行对应的特征提取与模型训练，最后得到测试集。
### 具体步骤与代码分析如下：
### 4.1.读取训练集与数据集:
```python
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
```
### 4.2.将时间列转换为时间戳:
```python
train_data['time'] = pd.to_datetime(train_data['time']).astype('int64') // 10**9
test_data['time'] = pd.to_datetime(test_data['time']).astype('int64') // 10**9
```
### 4.3.提取训练集和测试集特征和标签:
```python
X_train = train_data[['lat', 'lon', '速度', '方向', 'time']]
y_train = train_data['type']
X_test = test_data[['lat', 'lon', '速度', '方向', 'time']]
```
### 4.4.特征标准化:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
### 4.5.建立SVM模型并训练:
```python
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
```
### 4.6.将预测结果添加到测试集中并输出结果:
```python
test_data['type'] = y_pred
print(test_data)
output_path = 'predicted_test_data.csv'
test_data.to_csv(output_path, index=False)
```
### 其中使用较为重要的算法与模型有
### 1. 支持向量机（SVM）
支持向量机（Support Vector Machine, SVM）是一个强大的分类算法，能够用于线性和非线性数据的分类任务。在这段代码中，使用的是SVM中的支持向量分类器（SVC）。
#### 具体实现：
```python
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
```
-SVC：这是一个支持向量分类器。
-kernel=’rbf'：这里使用径向基函数（RBF）核，它是一种常用的核函数，适合非线性分类问题。
-C=1.0：惩罚参数C，它控制训练时对每个样本分类错误的惩罚力度。C越大，模型对误分类的惩罚越大，从而可能使模型更复杂。
-gamma=’scale’：核系数，默认为'scale'，表示使用1 / (n_features * X.var())作为gamma值。
-random_state=42：设置随机种子，以保证结果的可重复性。
### 2. 特征标准化
特征标准化是数据预处理中的一个重要步骤，目的是将不同特征的数据缩放到相同的尺度上，这对某些机器学习算法（包括SVM）来说是非常重要的。
#### 具体实现：
``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
-StandardScaler：这个类将特征值转换为均值为0、方差为1的标准正态分布。这可以提高模型的收敛速度和稳定性。
-fit_transform：对训练数据拟合并进行标准化。
-transform：使用在训练数据上学到的参数对测试数据进行标准化。
## 5、结果与分析
## 结果：
![image](https://github.com/sskaikai/my--blog-code/assets/165535013/35f7005e-41b8-46f9-89b5-bf91854a9b13)
## 分析：
这里主要使用了以下两个机器学习相关的算法/模型：
### 1.支持向量机（SVM）：用于分类任务。
### 2. 特征标准化（Standardization）：用于数据预处理。
这些步骤一起构成了一个典型的机器学习工作流程，从数据预处理到模型训练再到预测。通过标准化特征，确保每个特征对模型训练的影响是均衡的；通过使用支持向量机，可以有效地处理复杂的分类任务。
## 总结：
在这个实验中，我大致学会了利用机器学习相关技术，建立稳健的渔船作业方式识别模型，有效识别渔船的作业方式。虽然还有瑕疵，但会继续加油.
