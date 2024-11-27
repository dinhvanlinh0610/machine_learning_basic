import numpy as np

# Hàm hồi quy tuyến tính
def linear_regression(X, y, x_new=None):
    # Tính X^T * X
    XT_X = np.dot(X.T, X)
    
    # Tính nghịch đảo của (X^T * X)
    XT_X_inv = np.linalg.inv(XT_X)
    
    # Tính X^T * y
    XT_y = np.dot(X.T, y)
    
    # Tính w = (X^T * X)^-1 * X^T * y
    w = np.dot(XT_X_inv, XT_y)
    
    # Hiển thị kết quả
    print("Ma trận X^T * X:")
    print(XT_X)

    print("\nNghịch đảo của X^T * X:")
    print(XT_X_inv)

    print("\nMa trận X^T * y:")
    print(XT_y)

    print("\nTrọng số w (kết quả cuối cùng):")
    print(w)
    
    # Nếu có giá trị đầu vào mới, tính toán giá dự đoán
    if x_new is not None:
        x_new = np.array(x_new).reshape(1, -1)
        predicted_value = np.dot(x_new, w)
        print("\nGiá dự đoán cho x = {}:".format(x_new[0]))
        print(predicted_value[0, 0])
    
    return w

# Ví dụ sử dụng
X = np.array([
    [60, 2, 10],
    [40, 2, 5],
    [100, 3, 7]
])

y = np.array([10, 12, 20]).reshape(-1, 1)

# Gọi hàm để tính toán trọng số và dự đoán
w = linear_regression(X, y, x_new=[50, 2, 8])
