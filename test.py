import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk
from lightgbm import LGBMRegressor

# Đọc dữ liệu từ các tệp
data_nha = pd.read_csv("C:\\houseProject\\data_nha.csv")
data_new_chungcu = pd.read_csv("C:\\houseProject\\data_new_chungcu.csv")

# Ghép theo chiều dọc
data = pd.concat([data_nha, data_new_chungcu], ignore_index=True)

# Tiền xử lý dữ liệu
selected_columns = ["Quận", "Diện tích", "Số tầng", "Số phòng ngủ", "WC", "Thang Máy", "Mặt tiền", "Vị trí", "Kiểu nhà",
                    "Pháp lý", "Giá nhà(Tỷ)"]
new_data = data[selected_columns].copy()

# Loại bỏ các outlier
# diện tích
new_data['Lower Bound'] = new_data.groupby(['Quận'])['Diện tích'].transform(
    lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])['Diện tích'].transform(
    lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[
    (new_data['Diện tích'] >= new_data['Lower Bound']) & (new_data['Diện tích'] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# số tầng
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Số tầng"].transform(
    lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Số tầng"].transform(
    lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[(new_data["Số tầng"] >= new_data['Lower Bound']) & (new_data["Số tầng"] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# số phòng ngủ
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Số phòng ngủ"].transform(
    lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Số phòng ngủ"].transform(
    lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[
    (new_data["Số phòng ngủ"] >= new_data['Lower Bound']) & (new_data["Số phòng ngủ"] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# Mặt tiền
min_ = new_data["Mặt tiền"].quantile(0.01)
max_ = new_data["Mặt tiền"].quantile(0.99)
new_data = new_data[new_data["Mặt tiền"] >= min_]
new_data = new_data[new_data["Mặt tiền"] <= max_]

# WC
min_ = new_data["WC"].quantile(0.04)
max_ = new_data["WC"].quantile(0.99)
new_data = new_data[new_data["WC"] >= min_]
new_data = new_data[new_data["WC"] <= max_]

# Giá nhà
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Giá nhà(Tỷ)"].transform(
    lambda x: x.quantile(0.45) - 1.5 * (x.quantile(0.55) - x.quantile(0.45)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Giá nhà(Tỷ)"].transform(
    lambda x: x.quantile(0.55) + 1.5 * (x.quantile(0.55) - x.quantile(0.45)))
new_data = new_data[
    (new_data["Giá nhà(Tỷ)"] >= new_data['Lower Bound']) & (new_data["Giá nhà(Tỷ)"] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# ordinal_feature
# Quận
quan_values = {'Quận Đống Đa': 192, 'Quận Hai Bà Trưng': 214, 'Quận Hà Đông': 114, 'Quận Hoàn Kiếm': 579,
               'Quận Cầu Giấy': 214, ' Quận Cầu Giấy': 214, 'Quận Hoàng Mai': 112, 'Quận Bắc Từ Liêm': 91.7,
               'Quận Ba Đình': 207,
               'Quận Nam Từ Liêm': 113, 'Quận Tây Hồ': 213, 'Quận Thanh Xuân': 165, 'Quận Long Biên': 114}
new_data['Quan'] = new_data["Quận"].map(quan_values)

# Vị trí
vitri_values = {'nan': 1, 'Nhà hẻm,ngõ': 0, 'Nhà đường nội bộ,Cổ Nhuế': 1, 'Nhà mặt tiền,phố': 3, 'Biệt thự,liền kề': 2}
new_data['Vitri'] = new_data["Vị trí"].map(vitri_values)
new_data = new_data.drop(["Vị trí"], axis=1)

# Pháp lý
phaply_values = {'Sổ hồng': 0, 'Sổ đỏ': 2}
new_data['phaply'] = new_data["Pháp lý"].map(phaply_values)
new_data = new_data.drop(["Pháp lý"], axis=1)

# kieu nha
kieunha_values = {'Nhà': 2, 'Chung cư': 0}
new_data['kieunha'] = new_data["Kiểu nhà"].map(kieunha_values)
new_data = new_data.drop(["Kiểu nhà"], axis=1)

# Thiết lập biến mục tiêu và các biến đặc trưng
target = "Giá nhà(Tỷ)"
x = new_data.drop(target, axis=1)
y = new_data[target]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

# Tiền xử lý dữ liệu số
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", RobustScaler())
])

# Tiền xử lý dữ liệu nominal
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["Diện tích", "Số tầng", "Số phòng ngủ", "Mặt tiền", "Thang Máy", "WC", "Quan", "Vitri", "kieunha", "phaply"]),
])
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor())
]
)

# Huấn luyện mô hình stacking
reg.fit(x_train, y_train)

# Dự đoán giá nhà trên tập kiểm tra
y_predict_stacking = reg.predict(x_test)

def predict_price(input_data):

    predicted_price = reg.predict(input_data)
    r2 = r2_score(y_test, y_predict_stacking)
    mse = mean_squared_error(y_test, y_predict_stacking)
    mae = mean_absolute_error(y_test, y_predict_stacking)

    return predicted_price, r2, mse, mae

