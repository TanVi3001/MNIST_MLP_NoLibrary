📝 README.md
Nhận diện chữ số viết tay MNIST với Mạng truyền thẳng (MLP) thuần NumPy
Dự án này trình bày một triển khai đơn giản nhưng mạnh mẽ của mạng nơ-ron truyền thẳng (Multi-Layer Perceptron - MLP) để nhận diện chữ số viết tay từ tập dữ liệu MNIST. Điểm đặc biệt của dự án là toàn bộ quá trình xây dựng mô hình, từ khởi tạo trọng số, lan truyền tiến (Forward Propagation), lan truyền ngược (Backward Propagation) đến tối ưu hóa bằng Gradient Descent, đều được thực hiện chỉ bằng thư viện NumPy, không sử dụng các framework học sâu như TensorFlow hay PyTorch.

🚀 Các Tính năng Chính
Triển khai từ đầu với NumPy: Hiểu sâu sắc cách hoạt động của mạng nơ-ron mà không bị che khuất bởi các API cấp cao.
Tập dữ liệu MNIST: Sử dụng một trong những tập dữ liệu tiêu chuẩn nhất trong học máy cho bài toán phân loại ảnh.
Kiến trúc MLP 2 lớp: Bao gồm một lớp ẩn để trích xuất đặc trưng.
Hàm kích hoạt ReLU và Softmax: Áp dụng các hàm kích hoạt phổ biến trong mạng nơ-ron.
Mini-batch Gradient Descent: Phương pháp tối ưu hóa hiệu quả để cập nhật trọng số.
Đánh giá toàn diện: Sử dụng các chỉ số như Accuracy, Precision, Recall, F1-Score và Confusion Matrix.
Tiền xử lý ảnh tùy chỉnh: Chức năng xử lý ảnh đầu vào từ bên ngoài để chuẩn hóa về định dạng MNIST, cho phép dự đoán trên ảnh tự vẽ/chụp.
Lưu/Tải mô hình: Khả năng lưu và tải trọng số của mô hình đã huấn luyện.
🧠 Kiến trúc Mạng Nơ-ron (MLP)
Mạng nơ-ron được xây dựng trong dự án có kiến trúc 3 lớp:

Lớp Đầu vào (Input Layer): 784 nơ-ron (tương ứng với 28x28 pixel của ảnh MNIST đã được làm phẳng).
Lớp Ẩn (Hidden Layer): 128 nơ-ron, sử dụng hàm kích hoạt ReLU.
Lớp Đầu ra (Output Layer): 10 nơ-ron (tương ứng với 10 chữ số từ 0 đến 9), sử dụng hàm kích hoạt Softmax để đưa ra phân phối xác suất.
Công thức Toán học:
Lan truyền tiến (Forward Propagation):

Z1=X⋅W1+b1
A1=ReLU(Z1)
Z2=A1⋅W2+b2
A2=Softmax(Z2) (đây là xác suất dự đoán)
Lan truyền ngược (Backward Propagation):

dZ2=A2−Yone_hot (sai số)
dW2=1mAT1⋅dZ2
db2=1m∑hàngdZ2
dZ1=(dZ2⋅WT2)⊙ReLU′(Z1)
dW1=1mXT⋅dZ1
db1=1m∑hàngdZ1
Cập nhật Trọng số (Gradient Descent):

W=W−α⋅dW
b=b−α⋅db
Trong đó:

X: Dữ liệu đầu vào.
Yone_hot: Nhãn thực tế dưới dạng One-hot encoding.
W1,b1: Trọng số và Bias của lớp ẩn.
W2,b2: Trọng số và Bias của lớp đầu ra.
α: Tốc độ học (learning rate).
m: Số lượng mẫu trong một mini-batch.
🛠️ Cài đặt và Yêu cầu
Để chạy dự án này, bạn cần cài đặt các thư viện sau:

pip install numpy scikit-learn opencv-python Pillow scipy matplotlib seaborn
🚀 Cách Sử dụng
Tải dữ liệu MNIST: Dữ liệu MNIST được tải tự động bằng sklearn.datasets.fetch_openml.

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)
Tiền xử lý và Chia tập dữ liệu: Dữ liệu được chuẩn hóa về khoảng [0, 1] và chia thành tập huấn luyện (60.000 ảnh) và tập kiểm tra (10.000 ảnh).

X_normalized = X.astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=10000, random_state=42)
Huấn luyện Mô hình: Chạy hàm mini_batch_gradient_descent để huấn luyện mô hình.

W1_final, b1_final, W2_final, b2_final = mini_batch_gradient_descent(
    X_train, y_train, batch_size=64, alpha=0.1, epochs=20
)
Lưu và Tải Mô hình: Trọng số của mô hình được lưu vào file .npz và có thể tải lại để sử dụng sau này mà không cần huấn luyện lại.

model_filename = 'numpy_mnist_model.npz'
np.savez(model_filename, W1=W1_final, b1=b1_final, W2=W2_final, b2=b2_final)
# Để tải lại
loaded_data = np.load(model_filename)
W1_loaded = loaded_data['W1']
# ... và các trọng số khác
Dự đoán trên ảnh tùy chỉnh: Sử dụng hàm predict_my_digit để dự đoán chữ số từ một file ảnh.

predict_my_digit("path/to/your/image.png")
Lưu ý: Hàm này bao gồm các bước tiền xử lý ảnh phức tạp để biến ảnh đầu vào thành định dạng tương tự MNIST (28x28 pixel, căn giữa, khử nhiễu). Nó cũng bao gồm các bộ lọc hình học để cải thiện độ chính xác với một số trường hợp cụ thể (ví dụ: phân biệt số 1, 4, 7 hoặc 5, 6).

Đánh giá Hiệu năng: Sau khi huấn luyện, mô hình được đánh giá trên tập kiểm tra (X_test) bằng classification_report và confusion_matrix.

# Xem lại mã nguồn trong phần cuối của notebook để đánh giá hiệu năng
📊 Kết quả
Sau 20 epochs huấn luyện với mini-batch size = 64 và learning rate = 0.1, mô hình đạt được:

Độ chính xác trên tập Huấn luyện: Khoảng 99.54%
Độ chính xác trên tập Kiểm tra (10.000 ảnh): Khoảng 97.34%
Báo cáo phân loại (Classification Report)
(Ví dụ kết quả)

              precision    recall  f1-score   support

           0     0.9837    0.9817    0.9827       983
           1     0.9836    0.9887    0.9861      1152
           2     0.9565    0.9783    0.9673       967
           3     0.9643    0.9671    0.9657      1034
           4     0.9798    0.9658    0.9728       906
           5     0.9838    0.9712    0.9774       937
           6     0.9715    0.9948    0.9830       961
           7     0.9699    0.9763    0.9731      1055
           8     0.9705    0.9505    0.9604       969
           9     0.9706    0.9575    0.9640      1036

    accuracy                         0.9734     10000
   macro avg     0.9734    0.9732    0.9733     10000
weighted avg     0.9735    0.9734    0.9734     10000
Ma trận Nhầm lẫn (Confusion Matrix)
Ma trận nhầm lẫn trực quan hóa hiệu suất của mô hình, cho thấy số lượng mẫu được phân loại đúng và sai cho mỗi lớp.
