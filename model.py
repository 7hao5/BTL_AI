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

# Đọc dữ liệu từ các tệp
data_nha = pd.read_csv("C:\\houseProject\\data_nha.csv")
data_chungcu = pd.read_csv("C:\\houseProject\\data_chungcu.csv")
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
    ("num_feature", num_transformer, ["Diện tích", "Số tầng", "Số phòng ngủ", "Mặt tiền", "Thang Máy", "WC"]),
    ("nom_feature", nom_transformer, ["Quan", "Vitri", "Kiểu nhà", "phaply"]),
])

# Xây dựng các mô hình thành phần
xgb_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(random_state=43))
])

rf_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=43, n_estimators=200, criterion="absolute_error", max_depth=10))
])

lr_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Xây dựng mô hình stacking
estimators = [
    ('xgb', xgb_reg),
    ('rf', rf_reg),
    ('lr', lr_reg)
]

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

# Huấn luyện mô hình stacking
stacking_regressor.fit(x_train, y_train)

# Dự đoán giá nhà trên tập kiểm tra
y_predict_stacking = stacking_regressor.predict(x_test)

# Đánh giá mô hình
print("R2:", r2_score(y_test, y_predict_stacking))
print("MSE:", mean_squared_error(y_test, y_predict_stacking))
print("MAE:", mean_absolute_error(y_test, y_predict_stacking))


# Thiết lập giao diện người dùng
def predict_price():
    row = {}
    str_nha = nha_var.get()
    row["Diện tích"] = float(dien_tich_var.get())
    row["Số tầng"] = int(so_tang_var.get()) if str_nha == 'Nhà' else 1
    row["Số phòng ngủ"] = int(so_phong_ngu_var.get())
    row["WC"] = int(wc_var.get())
    row["Thang Máy"] = int(thang_may_var.get()) if str_nha == 'Nhà' else 0
    row["Mặt tiền"] = float(mat_tien_var.get()) if str_nha == 'Nhà' else 0
    row["Kiểu nhà"] = str_nha

    str1 = quan_var.get()
    if str1 in quan_values:
        row["Quan"] = quan_values[str1]
    str2 = vi_tri_var.get()
    if str2 in vitri_values:
        row["Vitri"] = vitri_values[str2]
    str3 = phap_ly_var.get()
    if str3 in phaply_values:
        row["phaply"] = phaply_values[str3]

    input_data = pd.DataFrame([row])
    predicted_price = stacking_regressor.predict(input_data)
    result_label.config(text=f"Giá trị dự đoán là: {predicted_price[0]:.2f} tỷ đồng")


# Hàm reset form
def reset_form():
    nha_var.set("")
    dien_tich_var.set("")
    so_tang_var.set("")
    so_phong_ngu_var.set("")
    wc_var.set("")
    thang_may_var.set("")
    mat_tien_var.set("")
    quan_var.set("")
    vi_tri_var.set("")
    phap_ly_var.set("")
    result_label.config(text="")


# Hàm thay đổi frame hiển thị
def show_frame(frame):
    frame.tkraise()


# Thiết lập giao diện người dùng
app = tk.Tk()
app.title("Dự đoán giá nhà")
app.geometry("800x600")
app.configure(bg='#dfe6e9')

# Sidebar
sidebar = tk.Frame(app, width=200, bg='#74b9ff', height=600, relief='sunken', borderwidth=2)
sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

home_button = tk.Button(sidebar, text="Home", font=('Arial', 12), command=lambda: show_frame(home_frame), bg='#74b9ff')
home_button.pack(fill='both')

nha_button = tk.Button(sidebar, text="Nhà", font=('Arial', 12), command=lambda: show_frame(nha_frame), bg='#74b9ff')
nha_button.pack(fill='both')

chungcu_button = tk.Button(sidebar, text="Chung cư", font=('Arial', 12), command=lambda: show_frame(chungcu_frame),
                           bg='#74b9ff')
chungcu_button.pack(fill='both')

# Main content frame
main_frame = tk.Frame(app, bg='#dfe6e9')
main_frame.pack(expand=True, fill='both', side='right')

# Home frame
home_frame = tk.Frame(main_frame, bg='#dfe6e9')
home_frame.grid(row=0, column=0, sticky='nsew')
home_label = ttk.Label(home_frame, text="Home", font=("Helvetica", 24), background='#dfe6e9')
home_label.pack(pady=20)

# Nhà frame
nha_frame = tk.Frame(main_frame, bg='#dfe6e9')
nha_frame.grid(row=0, column=0, sticky='nsew')

nha_label = ttk.Label(nha_frame, text="Dự đoán giá Nhà", font=("Helvetica", 24), background='#dfe6e9')
nha_label.grid(column=0, row=0, columnspan=2, pady=20)

nha_var = tk.StringVar()
dien_tich_var = tk.StringVar()
so_tang_var = tk.StringVar()
so_phong_ngu_var = tk.StringVar()
wc_var = tk.StringVar()
thang_may_var = tk.StringVar()
mat_tien_var = tk.StringVar()
quan_var = tk.StringVar()
vi_tri_var = tk.StringVar()
phap_ly_var = tk.StringVar()

ttk.Label(nha_frame, text="Diện tích:").grid(column=0, row=1, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=dien_tich_var).grid(column=1, row=1, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Số tầng:").grid(column=0, row=2, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=so_tang_var).grid(column=1, row=2, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Số phòng ngủ:").grid(column=0, row=3, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=so_phong_ngu_var).grid(column=1, row=3, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="WC:").grid(column=0, row=4, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=wc_var).grid(column=1, row=4, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Thang Máy:").grid(column=0, row=5, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=thang_may_var).grid(column=1, row=5, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Mặt tiền:").grid(column=0, row=6, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=mat_tien_var).grid(column=1, row=6, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Vị trí:").grid(column=0, row=7, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=vi_tri_var).grid(column=1, row=7, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Quận:").grid(column=0, row=8, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=quan_var).grid(column=1, row=8, padx=10, pady=5, sticky='we')

ttk.Label(nha_frame, text="Pháp lý:").grid(column=0, row=9, padx=10, pady=5, sticky='w')
ttk.Entry(nha_frame, textvariable=phap_ly_var).grid(column=1, row=9, padx=10, pady=5, sticky='we')

predict_button = ttk.Button(nha_frame, text="Dự đoán giá", command=predict_price)
predict_button.grid(column=0, row=10, columnspan=2, pady=10, sticky='we')

reset_button = ttk.Button(nha_frame, text="Reset", command=reset_form)
reset_button.grid(column=0, row=11, columnspan=2, pady=10, sticky='we')

result_label = ttk.Label(nha_frame, text="", font=('Arial', 14), foreground='green')
result_label.grid(column=0, row=12, columnspan=2, pady=20)

# Chung cư frame
chungcu_frame = tk.Frame(main_frame, bg='#dfe6e9')
chungcu_frame.grid(row=0, column=0, sticky='nsew')

chungcu_label = ttk.Label(chungcu_frame, text="Dự đoán giá Chung cư", font=("Helvetica", 24), background='#dfe6e9')
chungcu_label.grid(column=0, row=0, columnspan=2, pady=20)

dien_tich_cc_var = tk.StringVar()
so_phong_ngu_cc_var = tk.StringVar()
wc_cc_var = tk.StringVar()
quan_cc_var = tk.StringVar()
phap_ly_cc_var = tk.StringVar()

ttk.Label(chungcu_frame, text="Diện tích:").grid(column=0, row=1, padx=10, pady=5, sticky='w')
ttk.Entry(chungcu_frame, textvariable=dien_tich_cc_var).grid(column=1, row=1, padx=10, pady=5, sticky='we')

ttk.Label(chungcu_frame, text="Số phòng ngủ:").grid(column=0, row=2, padx=10, pady=5, sticky='w')
ttk.Entry(chungcu_frame, textvariable=so_phong_ngu_cc_var).grid(column=1, row=2, padx=10, pady=5, sticky='we')

ttk.Label(chungcu_frame, text="WC:").grid(column=0, row=3, padx=10, pady=5, sticky='w')
ttk.Entry(chungcu_frame, textvariable=wc_cc_var).grid(column=1, row=3, padx=10, pady=5, sticky='we')

ttk.Label(chungcu_frame, text="Quận:").grid(column=0, row=4, padx=10, pady=5, sticky='w')
ttk.Entry(chungcu_frame, textvariable=quan_cc_var).grid(column=1, row=4, padx=10, pady=5, sticky='we')

ttk.Label(chungcu_frame, text="Pháp lý:").grid(column=0, row=5, padx=10, pady=5, sticky='w')
ttk.Entry(chungcu_frame, textvariable=phap_ly_cc_var).grid(column=1, row=5, padx=10, pady=5, sticky='we')

predict_button_cc = ttk.Button(chungcu_frame, text="Dự đoán giá")
predict_button_cc.grid(column=0, row=6, columnspan=2, pady=10, sticky='we')

reset_button_cc = ttk.Button(chungcu_frame, text="Reset")
reset_button_cc.grid(column=0, row=7, columnspan=2, pady=10, sticky='we')

result_label_cc = ttk.Label(chungcu_frame, text="", font=('Arial', 14), foreground='green')
result_label_cc.grid(column=0, row=8, columnspan=2, pady=20)

# Cấu hình các cột để chúng có thể thay đổi kích thước
nha_frame.columnconfigure(0, weight=1)
nha_frame.columnconfigure(1, weight=1)
chungcu_frame.columnconfigure(0, weight=1)
chungcu_frame.columnconfigure(1, weight=1)

# Hiển thị frame mặc định
show_frame(home_frame)

# Chạy ứng dụng
app.mainloop()