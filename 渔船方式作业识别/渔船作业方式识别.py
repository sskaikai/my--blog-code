import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 读取训练集和测试集
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 将时间列转换为时间戳
train_data['time'] = pd.to_datetime(train_data['time']).astype('int64') // 10**9
test_data['time'] = pd.to_datetime(test_data['time']).astype('int64') // 10**9

# 提取训练集特征和标签
X_train = train_data[['lat', 'lon', '速度', '方向', 'time']]
y_train = train_data['type']

# 提取测试集特征
X_test = test_data[['lat', 'lon', '速度', '方向', 'time']]

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立SVM模型并训练
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = svm_model.predict(X_test)

# 将预测结果添加到测试集中
test_data['type'] = y_pred

# 输出带有预测结果的测试集
print(test_data)

# 保存带有预测结果的测试集到新的CSV文件
output_path = 'predicted_test_data.csv'
test_data.to_csv(output_path, index=False)