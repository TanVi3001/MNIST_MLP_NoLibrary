📝 Nhận diện chữ số viết tay MNIST với MLP thuần NumPy

Dự án này trình bày cách xây dựng một mạng nơ-ron truyền thẳng (Multi-Layer Perceptron - MLP) để nhận diện chữ số viết tay từ tập dữ liệu MNIST.

Điểm đặc biệt của dự án là toàn bộ quá trình xây dựng và huấn luyện mạng nơ-ron — từ khởi tạo trọng số, Forward Propagation, Backward Propagation đến Gradient Descent — đều được triển khai bằng NumPy, không sử dụng các framework Deep Learning như TensorFlow hoặc PyTorch.

🚀 Tính năng chính

Triển khai từ đầu với NumPy: Giúp hiểu rõ cơ chế hoạt động của mạng nơ-ron mà không phụ thuộc vào API cấp cao.

Tập dữ liệu MNIST: Sử dụng bộ dữ liệu tiêu chuẩn cho bài toán nhận diện chữ số viết tay.

Kiến trúc MLP 2 lớp: Bao gồm một lớp ẩn và một lớp đầu ra.

ReLU và Softmax: Sử dụng ReLU cho hidden layer và Softmax cho output layer.

Mini-batch Gradient Descent: Huấn luyện theo từng mini-batch để tối ưu hiệu quả.

Đánh giá mô hình: Sử dụng Accuracy, Precision, Recall, F1-Score và Confusion Matrix.

Dự đoán ảnh tùy chỉnh: Cho phép đưa ảnh chữ số bên ngoài vào mô hình.

Tiền xử lý ảnh: Chuẩn hóa ảnh đầu vào về định dạng tương tự MNIST.

Lưu và tải mô hình: Lưu các trọng số đã huấn luyện dưới dạng .npz.

🧠 Kiến trúc mạng nơ-ron

Mô hình sử dụng kiến trúc MLP gồm 3 tầng:

Input Layer          Hidden Layer          Output Layer
784 neurons    →     128 neurons      →     10 neurons
28 × 28              ReLU                   Softmax

Chi tiết

Layer

Số neuron

Activation

Mục đích

Input

784

-

Ảnh MNIST 28 × 28 được flatten

Hidden

128

ReLU

Học các đặc trưng của ảnh

Output

10

Softmax

Xác suất của các chữ số 0–9

Input Layer

Mỗi ảnh MNIST có kích thước:

28 × 28 = 784 pixels

Ảnh được flatten thành vector:

X \in \mathbb{R}^{784}

Hidden Layer

Hidden layer gồm 128 neuron và sử dụng hàm kích hoạt ReLU:

ReLU(x) = max(0, x)

Output Layer

Output layer gồm 10 neuron, tương ứng với 10 chữ số:

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Hàm Softmax biến output thành phân phối xác suất:

P(y = k | x)

📐 Công thức toán học

1. Forward Propagation

Đầu tiên, dữ liệu đi qua hidden layer:

$$Z_1 = XW_1 + b_1$$

Sau đó áp dụng ReLU:

$$A_1 = \mathrm{ReLU}(Z_1)$$

Tiếp theo, dữ liệu được truyền đến output layer:

$$Z_2 = A_1W_2 + b_2$$

Cuối cùng, Softmax tạo ra xác suất dự đoán:

$$A_2 = \mathrm{Softmax}(Z_2)$$

Trong đó:

$$A_2 = [P(0), P(1), ..., P(9)]$$

Nhãn dự đoán được xác định bằng:

$$\hat{y} = \mathrm{argmax}(A_2)$$

2. Backward Propagation

Với Softmax kết hợp Cross-Entropy Loss:

$$dZ_2 = A_2 - Y_{\text{one-hot}}$$

Gradient của trọng số output layer:

$$dW_2 = \frac{1}{m}A_1^T dZ_2$$

Gradient của bias output layer:

$$db_2 = \frac{1}{m}\sum dZ_2$$

Lan truyền gradient về hidden layer:

$$dZ_1 = (dZ_2W_2^T) \odot \mathrm{ReLU}'(Z_1)$$

Gradient của trọng số hidden layer:

$$dW_1 = \frac{1}{m}X^T dZ_1$$

Gradient của bias hidden layer:

$$db_1 = \frac{1}{m}\sum dZ_1$$

3. Gradient Descent

Các tham số được cập nhật theo công thức:

$$W := W - \alpha dW$$

$$b := b - \alpha db$$

Trong đó:

Ký hiệu

Ý nghĩa

$X$

Dữ liệu đầu vào

$Y_{\text{one-hot}}$

Nhãn thực tế dạng One-Hot

$W_1, b_1$

Trọng số và bias của hidden layer

$W_2, b_2$

Trọng số và bias của output layer

$\alpha$

Learning rate

$m$

Kích thước mini-batch

🛠️ Cài đặt

Yêu cầu

Python 3.x

NumPy

Scikit-learn

OpenCV

Pillow

SciPy

Matplotlib

Seaborn

Cài đặt thư viện

pip install numpy scikit-learn opencv-python Pillow scipy matplotlib seaborn

Hoặc nếu repository có file requirements.txt:

pip install -r requirements.txt

💻 Cách sử dụng

1. Tải dữ liệu MNIST

Dữ liệu MNIST được tải tự động thông qua fetch_openml:

from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    "mnist_784",
    version=1,
    cache=True,
    parser="auto"
)

X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)

2. Chuẩn hóa dữ liệu

Pixel của ảnh MNIST có giá trị từ 0 đến 255.

Dữ liệu được chuẩn hóa về khoảng [0, 1]:

X_normalized = X.astype("float32") / 255.0

Sau đó chia dữ liệu thành tập train và test:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    y,
    test_size=10000,
    random_state=42
)

Kết quả:

Training set : 60,000 images
Testing set  : 10,000 images

3. Huấn luyện mô hình

Mô hình được huấn luyện bằng Mini-batch Gradient Descent:

W1_final, b1_final, W2_final, b2_final = mini_batch_gradient_descent(
    X_train,
    y_train,
    batch_size=64,
    alpha=0.1,
    epochs=20
)

Thông số huấn luyện:

Parameter

Value

Epochs

20

Batch size

64

Learning rate

0.1

Hidden neurons

128

Output classes

10

💾 Lưu và tải mô hình

Sau khi huấn luyện, các trọng số có thể được lưu vào file .npz:

model_filename = "numpy_mnist_model.npz"

np.savez(
    model_filename,
    W1=W1_final,
    b1=b1_final,
    W2=W2_final,
    b2=b2_final
)

Tải lại mô hình

loaded_data = np.load("numpy_mnist_model.npz")

W1_loaded = loaded_data["W1"]
b1_loaded = loaded_data["b1"]
W2_loaded = loaded_data["W2"]
b2_loaded = loaded_data["b2"]

Việc này cho phép sử dụng mô hình đã huấn luyện mà không cần train lại từ đầu.

✍️ Dự đoán chữ số tùy chỉnh

Mô hình hỗ trợ dự đoán chữ số từ ảnh bên ngoài:

predict_my_digit("path/to/your/image.png")

Hàm predict_my_digit() thực hiện các bước tiền xử lý cần thiết để đưa ảnh về dạng tương tự MNIST:

Input Image
     │
     ▼
Grayscale
     │
     ▼
Noise Reduction
     │
     ▼
Thresholding
     │
     ▼
Crop Digit
     │
     ▼
Resize
     │
     ▼
Centering
     │
     ▼
28 × 28
     │
     ▼
Normalize [0, 1]
     │
     ▼
MLP Prediction

Các bước tiền xử lý giúp mô hình xử lý tốt hơn đối với ảnh chữ số được tự vẽ hoặc chụp từ bên ngoài.

Lưu ý: Ảnh tùy chỉnh có thể cho kết quả thấp hơn MNIST test set do khác biệt về nét chữ, độ dày nét, background, vị trí chữ số và chất lượng ảnh.

📊 Đánh giá mô hình

Sau khi huấn luyện, mô hình được đánh giá trên tập test bằng:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Ví dụ:

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

📈 Kết quả

Sau 20 epochs với:

Batch size    = 64
Learning rate = 0.1

mô hình đạt được kết quả xấp xỉ:

Dataset

Accuracy

Training

99.54%

Testing

97.34%

Classification Report

Digit

Precision

Recall

F1-Score

Support

0

0.9837

0.9817

0.9827

983

1

0.9836

0.9887

0.9861

1152

2

0.9565

0.9783

0.9673

967

3

0.9643

0.9671

0.9657

1034

4

0.9798

0.9658

0.9728

906

5

0.9838

0.9712

0.9774

937

6

0.9715

0.9948

0.9830

961

7

0.9699

0.9763

0.9731

1055

8

0.9705

0.9505

0.9604

969

9

0.9706

0.9575

0.9640

1036

Accuracy





0.9734

10000

Macro Avg

0.9734

0.9732

0.9733

10000

Weighted Avg

0.9735

0.9734

0.9734

10000

🔲 Confusion Matrix

Confusion Matrix giúp quan sát trực quan các trường hợp mô hình dự đoán đúng hoặc nhầm giữa các chữ số.

Ví dụ:

             Predicted
          0  1  2  3  ... 9
Actual 0  ■  ·  ·  ·  ... ·
       1  ·  ■  ·  ·  ... ·
       2  ·  ·  ■  ·  ... ·
       3  ·  ·  ·  ■  ... ·
       ...
       9  ·  ·  ·  ·  ... ■

Nếu repository có ảnh Confusion Matrix, nên lưu ảnh vào:

assets/confusion_matrix.png

Sau đó nhúng vào README:

![Confusion Matrix](assets/confusion_matrix.png)

📁 Cấu trúc dự án đề xuất

numpy-mnist-mlp/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── notebooks/
│   └── mnist_mlp_numpy.ipynb
│
├── models/
│   └── numpy_mnist_model.npz
│
├── assets/
│   └── confusion_matrix.png
│
└── examples/
    └── digit.png

Nếu project của bạn đang có cấu trúc thư mục khác, hãy thay phần trên bằng cấu trúc thực tế của repository.

🧩 Công nghệ sử dụng

Python

NumPy — triển khai mạng nơ-ron và phép toán ma trận

Scikit-learn — tải dữ liệu và đánh giá mô hình

OpenCV — xử lý ảnh

Pillow — đọc và xử lý ảnh

SciPy — các phép toán khoa học

Matplotlib — trực quan hóa

Seaborn — trực quan hóa Confusion Matrix

🎯 Mục tiêu học tập

Dự án được thực hiện nhằm hiểu rõ các thành phần cơ bản của Neural Network:

Input
  ↓
Linear Transformation
  ↓
ReLU
  ↓
Linear Transformation
  ↓
Softmax
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradient Descent
  ↓
Updated Weights

Thay vì sử dụng trực tiếp các framework như TensorFlow hoặc PyTorch, dự án triển khai các thành phần cốt lõi bằng NumPy để hiểu rõ hơn về cơ chế hoạt động bên trong của mạng nơ-ron.

📌 Hạn chế

Mặc dù đạt khoảng 97.34% accuracy trên MNIST test set, mô hình vẫn có một số hạn chế:

MLP không khai thác trực tiếp cấu trúc không gian của ảnh.

Không có Convolution như CNN.

Khả năng tổng quát hóa với ảnh ngoài MNIST có thể thấp hơn.

Tiền xử lý ảnh tùy chỉnh có ảnh hưởng đáng kể đến kết quả.

Mô hình chưa sử dụng các kỹ thuật tối ưu hiện đại như Adam, Batch Normalization hoặc Dropout.

🚀 Hướng phát triển

Một số hướng có thể mở rộng dự án:

Thêm CNN thuần NumPy

So sánh MLP với CNN

Thử nghiệm các learning rate khác nhau

Thử nghiệm các kích thước hidden layer khác nhau

Implement Adam Optimizer từ đầu

Implement Dropout từ đầu

Thêm visualization cho quá trình training

Vẽ Learning Curve

Cải thiện preprocessing cho ảnh viết tay bên ngoài

Xây dựng giao diện Web Demo bằng Streamlit

So sánh mô hình NumPy với PyTorch/TensorFlow

📚 Dataset

Dự án sử dụng MNIST Handwritten Digit Dataset, gồm các ảnh chữ số viết tay có kích thước 28 × 28 pixels và 10 lớp tương ứng với các chữ số từ 0 đến 9.

👨‍💻 Tác giả

Lê Tấn Vĩ

Dự án được thực hiện nhằm phục vụ mục đích học tập và nghiên cứu về:

Machine Learning

Neural Networks

Deep Learning

Computer Vision

NumPy implementation from scratch

⭐ Nếu project hữu ích

Nếu repository này giúp bạn hiểu rõ hơn về cách một MLP hoạt động bên trong, hãy Star ⭐ repository để ủng hộ dự án.
