import tkinter as tk
from tkinter import ttk
import pandas as pd

# Đọc dữ liệu từ các tệp
data_nha = pd.read_csv("C:\\houseProject\\data_nha.csv")
data_new_chungcu = pd.read_csv("C:\\houseProject\\data_new_chungcu.csv")

# Ghép theo chiều dọc
data = pd.concat([data_nha, data_new_chungcu], ignore_index=True)

# Tiền xử lý dữ liệu
selected_columns = ["Quận", "Diện tích", "Số tầng", "Số phòng ngủ", "WC", "Thang Máy", "Mặt tiền", "Vị trí", "Kiểu nhà",
                    "Pháp lý", "Giá nhà(Tỷ)"]
new_data = data[selected_columns].copy()

# Hàm tính điểm tương đồng
def calculate_similarity(row1, row2):
    score = 0
    score += 10 * (row1["kieunha"] == row2["Kiểu nhà"])
    score += 10 * (row1["Quan"] == row2["Quận"])
    score += 5 * (row1["Vitri"] == row2["Vị trí"])
    score += 6 * (row1["phaply"] == row2["Pháp lý"])
    score += 4 * (abs(row1["Diện tích"] - row2["Diện tích"]) <= 10)
    score += 3 * (abs(row1["Số tầng"] - row2["Số tầng"]) <= 1)
    score += 1 * (abs(row1["Số phòng ngủ"] - row2["Số phòng ngủ"]) <= 1)
    score += 1 * (abs(row1["WC"] - row2["WC"]) <= 1)
    score += 1 * (abs(row1["Thang Máy"] - row2["Thang Máy"]) <= 1)
    score += 5 * (abs(row1["Mặt tiền"] - row2["Mặt tiền"]) <= 1)
    return score

def find_similar_houses(target_row):

    quan_value = target_row["Quan"].iloc[0]  # Lấy giá trị Quận từ target_row
    vitri_value = target_row["Vitri"].iloc[0]  # Lấy giá trị Vị trí từ target_row
    phaply_value = target_row["phaply"].iloc[0]  # Lấy giá trị pháp lý từ target_row

    # Tìm các bản ghi trong data thỏa mãn các điều kiện
    filtered_data = new_data[(new_data['Quận'] == quan_value) & (new_data['Vị trí'] == vitri_value)
                             & (new_data['Pháp lý'] == phaply_value)]

    # Tính toán điểm tương đồng cho các bản ghi đã lọc
    filtered_data["similarity"] = filtered_data.apply(lambda row: calculate_similarity(target_row, row), axis=1)

    # Sắp xếp và lấy 10 căn nhà có điểm tương đồng cao nhất
    return filtered_data.sort_values(by="similarity", ascending=False).head(30)

# # Tạo DataFrame đầu vào
# row = {
#     "Diện tích": float(50),
#     "Số tầng": int(1),
#     "Số phòng ngủ": int(2),
#     "WC": int(1),
#     "Thang Máy": int(0),
#     "Mặt tiền": float(0),
#     "kieunha": 'Chung cư',
#     "Quan": "Quận Hoàng Mai",
#     "Vitri": 'Nhà mặt tiền,phố',
#     "phaply": "Sổ đỏ"
# }
# target_row = pd.DataFrame([row])
# x = find_similar_houses(target_row)
# filtered_results = x[abs(x["Giá nhà(Tỷ)"] - 4) <= 1.5]
