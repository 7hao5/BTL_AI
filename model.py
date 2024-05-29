import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import OneHotEncoder

data_nha = pd.read_csv("data_nha.csv")
data_chungcu = pd.read_csv("data_chungcu.csv")
data_new_chungcu = pd.read_csv("data_new_chungcu.csv")

# Ghép theo chiều dọc
data = pd.concat([data_nha, data_new_chungcu], ignore_index=True)

# loai bo 1 so cot khong caan thiet
selected_columns = ["Quận", "Diện tích", "Số tầng", "Số phòng ngủ", "WC", "Thang Máy", "Mặt tiền", "Vị trí", "Kiểu nhà", "Pháp lý", "Giá nhà(Tỷ)"]
new_data = data[selected_columns].copy()

# loai bo cac outlier

# diện tích
new_data['Lower Bound'] = new_data.groupby(['Quận'])['Diện tích'].transform(lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])['Diện tích'].transform(lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[(new_data['Diện tích'] >= new_data['Lower Bound']) & (new_data['Diện tích'] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# số tầng
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Số tầng"].transform(lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Số tầng"].transform(lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[(new_data["Số tầng"] >= new_data['Lower Bound']) & (new_data["Số tầng"] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# số phòng ngủ
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Số phòng ngủ"].transform(lambda x: x.quantile(0.35) - 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Số phòng ngủ"].transform(lambda x: x.quantile(0.65) + 1.5 * (x.quantile(0.65) - x.quantile(0.35)))
new_data = new_data[(new_data["Số phòng ngủ"] >= new_data['Lower Bound']) & (new_data["Số phòng ngủ"] <= new_data['Upper Bound'])]
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
new_data['Lower Bound'] = new_data.groupby(['Quận'])["Giá nhà(Tỷ)"].transform(lambda x: x.quantile(0.45) - 1.5 * (x.quantile(0.55) - x.quantile(0.45)))
new_data['Upper Bound'] = new_data.groupby(['Quận'])["Giá nhà(Tỷ)"].transform(lambda x: x.quantile(0.55) + 1.5 * (x.quantile(0.55) - x.quantile(0.45)))
new_data = new_data[(new_data["Giá nhà(Tỷ)"] >= new_data['Lower Bound']) & (new_data["Giá nhà(Tỷ)"] <= new_data['Upper Bound'])]
new_data = new_data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# ordinal_feature
# Quận
quan_values = {'Quận Đống Đa': 192, 'Quận Hai Bà Trưng': 214, 'Quận Hà Đông': 114, 'Quận Hoàn Kiếm': 579,
               'Quận Cầu Giấy': 214, ' Quận Cầu Giấy': 214, 'Quận Hoàng Mai': 112, 'Quận Bắc Từ Liêm': 91.7, 'Quận Ba Đình': 207,
               'Quận Nam Từ Liêm': 113, 'Quận Tây Hồ': 213, 'Quận Thanh Xuân': 165, 'Quận Long Biên': 114}
new_data['Quan'] = new_data["Quận"].map(quan_values)
new_data = new_data.drop(["Quận"], axis=1)

# Vị trí
vitri_values = {'nan': 1, 'Nhà hẻm,ngõ': 0, 'Nhà đường nội bộ,Cổ Nhuế': 1, 'Nhà mặt tiền,phố': 3, 'Biệt thự,liền kề': 2}
new_data['Vitri'] = new_data["Vị trí"].map(vitri_values)
new_data = new_data.drop(["Vị trí"], axis=1)

# Pháp lý
phaply_values = {'Sổ hồng': 0, 'Sổ đỏ': 2}
new_data['phaply'] = new_data["Pháp lý"].map(phaply_values)
new_data = new_data.drop(["Pháp lý"], axis=1)

# feature vs target
target = "Giá nhà(Tỷ)"
x = new_data.drop(target, axis=1)
y = new_data[target]

# phan chia bo du lieu (train, test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

# xu ly du lieu dang so
# xu ly du lieu bi missingvalue, dua cac feature ve cung range
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", RobustScaler())
])
# xu ly du lieu nominal_feature
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=True))
])
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["Diện tích", "Số tầng", "Số phòng ngủ", "Mặt tiền", "Quan", "Thang Máy", "WC"]),
    ("nom_feature", nom_transformer, ["Kiểu nhà"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

# Tìm parameter tốt nhất cho LGBMRegressor
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
    "regressor__max_depth": [10, 20, 30],
    "regressor__n_estimators": [50, 100, 200],
}
reg_cv = GridSearchCV(reg, param_grid=param_grid, scoring="neg_mean_absolute_error", verbose=2, cv=6, n_jobs=8)

# train model
reg_cv.fit(x_train, y_train)
y_predict = reg_cv.predict(x_test)

for i, j in zip(y_test, y_predict):
    print("Thuc te: {}. Du doan: {}" .format(i, j))

# sử dụng metric để đánh giá
print("R2: {}" .format(r2_score(y_test, y_predict)))
print("MSE: {}" .format(mean_squared_error(y_test, y_predict)))
print("MAE: {}" .format(mean_absolute_error(y_test, y_predict)))

# hiển thị parameter tốt nhất của model
print(reg_cv.best_params_)

# Nhập từ người dùng
data2 = []
row = {}

str_nha = input("Nhập kiểu nhà (Chung cư, nhà): ")
if(str_nha == 'Nhà'):

    row["Diện tích"] = float(input("Nhập diện tích của căn nhà: "))
    row["Số tầng"] = int(input("Nhập số tầng của căn nhà: "))
    row["Số phòng ngủ"] = int(input("Nhập số phòng ngủ của căn nhà: "))
    row["WC"] = int(input("Nhập số WC của căn nhà: "))
    row["Thang Máy"] = int(input("Nhập số lượng thang máy của căn nhà: "))
    row["Mặt tiền"] = float(input("Nhập độ rộng mặt tiền của căn nhà: "))
    row["Kiểu nhà"] = str_nha

    str1 = input("Quận: ")
    if str1 in quan_values:
        row["Quan"] = quan_values[str1]
    str2 = input("Nhập vị trí (mặt phố, ngõ hẻm, ...) của căn nhà: ")
    if str2 in vitri_values:
        row["Vitri"] = vitri_values[str2]
    str3 = input("Nhập giấy tờ(sổ đỏ, sổ hồng) của căn nhà: ")
    if str3 in phaply_values:
        row["phaply"] = phaply_values[str3]
else:

    row["Diện tích"] = float(input("Nhập diện tích của căn nhà: "))
    row["Số tầng"] = int(1)
    row["Số phòng ngủ"] = int(input("Nhập số phòng ngủ của căn nhà: "))
    row["WC"] = int(input("Nhập số WC của căn nhà: "))
    row["Thang Máy"] = int(0)
    row["Mặt tiền"] = float(0)
    row["Kiểu nhà"] = str_nha

    str1 = input("Quận: ")
    if str1 in quan_values:
        row["Quan"] = quan_values[str1]
    str2 = input("Nhập vị trí (mặt phố, ngõ hẻm, ...) của căn nhà: ")
    if str2 in vitri_values:
        row["Vitri"] = vitri_values[str2]
    str3 = input("Nhập giấy tờ(sổ đỏ, sổ hồng) của căn nhà: ")
    if str3 in phaply_values:
        row["phaply"] = phaply_values[str3]

data2.append(row)
df = pd.DataFrame(data2)
df.info()
print("Giá trị dự đoán là: {}" .format(reg_cv.predict(df)))

# # tìm, chọn model tốt nhất
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(predictions)
