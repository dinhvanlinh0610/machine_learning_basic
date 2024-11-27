import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Hàm hồi quy tuyến tính với và không có bias
def linear_regression(X, y, x_new=None, fit_intercept=True):
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    
    # Hiển thị trọng số (w) và bias (intercept)
    print("\nTrọng số w (kết quả cuối cùng):")
    print(model.coef_)
    if fit_intercept:
        print("Bias (intercept):")
        print(model.intercept_)
    
    # Nếu có giá trị đầu vào mới, tính toán giá dự đoán
    if x_new is not None:
        x_new = np.array(x_new).reshape(1, -1)
        predicted_value = model.predict(x_new)
        print("\nGiá dự đoán cho x_new = {}:".format(x_new[0]))
        print(predicted_value[0])
    
    return model.coef_, model.intercept_ if fit_intercept else model.coef_

# Hàm nhập dữ liệu mẫu
def get_input_data():
    # Sử dụng pandas để nhập và xử lý dữ liệu
    data = pd.read_csv('data.csv')  # Giả sử dữ liệu được lưu trong file 'data.csv'

    # Hiển thị thông tin dữ liệu
    print("Dữ liệu mẫu:")
    print(data.head())
    
    # Chọn các cột để làm đặc trưng và nhãn
    feature_columns = input("Nhập các cột làm đặc trưng, cách nhau dấu phẩy (ví dụ: 'Feature1,Feature2'): ").split(',')
    label_column = input("Nhập tên cột nhãn: ")
    
    # Tạo ma trận X và nhãn y từ các cột đã chọn
    X = data[feature_columns].values
    y = data[label_column].values.reshape(-1, 1)
    
    # Nhập giá trị đầu vào mới (x_new) nếu có
    x_new = input("Nhập giá trị x_new (các giá trị cách nhau bằng dấu cách) hoặc nhấn Enter nếu không có: ")
    if x_new:  # Nếu có giá trị x_new
        x_new = list(map(float, x_new.split()))
    else:
        x_new = None

    return X, y, x_new

# Hàm hiển thị các biểu đồ
def plot_graphs(X, y, model):
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ 1: Dữ liệu gốc
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y, color='blue', label='Dữ liệu gốc')
    plt.xlabel('Feature 1')
    plt.ylabel('Y')
    plt.title('Dữ liệu gốc')
    plt.legend()

    # Biểu đồ 2: Dự đoán từ mô hình
    plt.subplot(1, 2, 2)
    y_pred = model.predict(X)
    plt.scatter(X[:, 0], y, color='blue', label='Dữ liệu gốc')
    plt.plot(X[:, 0], y_pred, color='red', label='Dự đoán từ mô hình')
    plt.xlabel('Feature 1')
    plt.ylabel('Y')
    plt.title('Dự đoán từ mô hình')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Hàm hiển thị lỗi trung bình bình phương (MSE)
def calculate_mse(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"\nLỗi trung bình bình phương (MSE): {mse}")

# Ví dụ sử dụng
X, y, x_new = get_input_data()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gọi hàm hồi quy tuyến tính
print("Hồi quy tuyến tính với bias:")
w_with_bias, bias = linear_regression(X_train, y_train, x_new=x_new, fit_intercept=True)

print("\nHồi quy tuyến tính không có bias:")
w_without_bias = linear_regression(X_train, y_train, x_new=x_new, fit_intercept=False)

# Tính toán và hiển thị MSE cho mô hình với bias
model_with_bias = LinearRegression(fit_intercept=True)
model_with_bias.fit(X_train, y_train)
calculate_mse(X_test, y_test, model_with_bias)

# Hiển thị biểu đồ
plot_graphs(X_train, y_train, model_with_bias)
