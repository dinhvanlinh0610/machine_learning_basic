import numpy as np

# Dữ liệu từ bài toán
X = np.array([
    [60, 2, 10],
    [40, 2, 5],
    [100, 3, 7]
])

y = np.array([10, 12, 20]).reshape(-1, 1)

# 1. Tính X^T * X
XT_X = np.dot(X.T, X)

# 2. Tính nghịch đảo của (X^T * X)
XT_X_inv = np.linalg.inv(XT_X)

# 3. Tính X^T * y
XT_y = np.dot(X.T, y)

# 4. Tính w = (X^T * X)^-1 * X^T * y
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

# Dự đoán giá nhà cho x = (50, 2, 8)
x_new = np.array([50, 2, 8]).reshape(1, -1)
predicted_price = np.dot(x_new, w)

print("\nGiá dự đoán cho căn nhà x = (50, 2, 8):")
print(predicted_price[0, 0])
