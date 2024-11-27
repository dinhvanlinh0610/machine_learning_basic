import numpy as np

# Hàm hồi quy tuyến tính tổng quát
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

# Hàm hồi quy tuyến tính với bias (bias trick)
def linear_regression_with_bias(X, y, x_new=None):
    # Bổ sung cột bias (1) vào ma trận X
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Tính (X^T * X)
    XT_X = np.dot(X_bias.T, X_bias)
    
    # Tính nghịch đảo của (X^T * X)
    XT_X_inv = np.linalg.inv(XT_X)
    
    # Tính (X^T * y)
    XT_y = np.dot(X_bias.T, y)
    
    # Tính w = (X^T * X)^-1 * X^T * y
    w = np.dot(XT_X_inv, XT_y)
    
    # Hiển thị kết quả
    print("Ma trận X^T * X (với bias):")
    print(XT_X)

    print("\nNghịch đảo của X^T * X (với bias):")
    print(XT_X_inv)

    print("\nMa trận X^T * y (với bias):")
    print(XT_y)

    print("\nTrọng số w (kết quả cuối cùng, bao gồm bias):")
    print(w)
    
    # Nếu có giá trị đầu vào mới, tính toán giá dự đoán
    if x_new is not None:
        x_new = np.array(x_new).reshape(1, -1)
        # Thêm bias vào x_new
        x_new_bias = np.c_[np.ones(x_new.shape[0]), x_new]
        predicted_value = np.dot(x_new_bias, w)
        print("\nGiá dự đoán cho x_new = {}:".format(x_new[0]))
        print(predicted_value[0, 0])
    
    return w

# Hàm nhập dữ liệu từ người dùng
def get_input_data():
    # Nhập số lượng mẫu (số dòng của X)
    n_samples = int(input("Nhập số lượng mẫu (số dòng của X): "))
    
    # Nhập số lượng đặc trưng (số cột của X)
    n_features = int(input("Nhập số lượng đặc trưng (số cột của X): "))
    
    # Nhập ma trận X
    print("Nhập ma trận đặc trưng X:")
    X = []
    for i in range(n_samples):
        row = list(map(float, input(f"Nhập dòng {i+1} của X (các giá trị cách nhau bằng dấu cách): ").split()))
        X.append(row)
    
    X = np.array(X)

    # Nhập giá trị y
    print("Nhập giá trị y (mục tiêu):")
    y = list(map(float, input("Nhập các giá trị y (các giá trị cách nhau bằng dấu cách): ").split()))
    y = np.array(y).reshape(-1, 1)
    
    # Nhập giá trị đầu vào mới (x_new) nếu có
    x_new = input("Nhập giá trị x_new (các giá trị cách nhau bằng dấu cách) hoặc nhấn Enter nếu không có: ")
    
    if x_new:  # Nếu người dùng nhập vào x_new
        x_new = list(map(float, x_new.split()))
    else:
        x_new = None
    
    return X, y, x_new

if __name__ == "__main__":
    # Nhập dữ liệu từ người dùng
    X, y, x_new = get_input_data()
    
    # Chọn cách tính toán (có bias hoặc không)
    has_bias = input("Có sử dụng bias không? (1.có/2.không): ")
    
    if has_bias.lower() == "1":
        w = linear_regression_with_bias(X, y, x_new)
    else:
        w = linear_regression(X, y, x_new)
