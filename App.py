import tkinter as tk
import tkinter as tk
from tkinter import ttk
import pandas as pd
from test import *
from nha_tuongDong import *
from chungCu_tuongDong import *
# from model import *

# Danh sách các quận tại Hà Nội
districts = [
    "Quận Đống Đa", "Quận Hai Bà Trưng", "Quận Hà Đông", "Quận Hoàn Kiếm", "Quận Cầu Giấy", "Quận Hoàng Mai", "Quận Bắc Từ Liêm",
    "Quận Ba Đình", "Quận Tây Hồ", "Quận Thanh Xuân", "Quận Hai Bà Trưng", "Quận Hà Đông", "Quận Hoàn Kiếm", "Quận Cầu Giấy",
    "Quận Hoàng Mai", "Quận Bắc Từ Liêm", "Quận Ba Đình", "Quận Tây Hồ", "Quận Thanh Xuân"
]

# Pháp lý
phapLy = [
    "Sổ hồng", "Sổ đỏ"
]
# Vị trí
viTri = [
    "Nhà hẻm,ngõ", "Nhà đường nội bộ,Cổ Nhuế", "Nhà mặt tiền,phố", "Biệt thự,liền kề"
]

# ma hoa quan
quan_values = {'Quận Đống Đa': 192, 'Quận Hai Bà Trưng': 214, 'Quận Hà Đông': 114, 'Quận Hoàn Kiếm': 579,
               'Quận Cầu Giấy': 214, ' Quận Cầu Giấy': 214, 'Quận Hoàng Mai': 112, 'Quận Bắc Từ Liêm': 91.7,
               'Quận Ba Đình': 207,
               'Quận Nam Từ Liêm': 113, 'Quận Tây Hồ': 213, 'Quận Thanh Xuân': 165, 'Quận Long Biên': 114}

# ma hoa phap ly
phaply_values = {'Sổ hồng': 0, 'Sổ đỏ': 2}

# ma hoa vi tri
vitri_values = {'nan': 1, 'Nhà hẻm,ngõ': 0, 'Nhà đường nội bộ,Cổ Nhuế': 1, 'Nhà mặt tiền,phố': 3, 'Biệt thự,liền kề': 2}

# Định nghĩa các thuật ngữ và giải thích
terms_explanation = {
    "Diện tích": "Diện tích căn nhà tính bằng mét vuông (m²).",
    "Số tầng": "Số tầng của căn nhà.",
    "Số phòng ngủ": "Số phòng ngủ trong căn nhà.",
    "WC": "Số phòng vệ sinh (WC) trong căn nhà.",
    "Thang Máy": "Số lượng thang máy trong căn nhà.",
    "Mặt tiền": "Chiều rộng mặt tiền của căn nhà tính bằng mét (m).",
    "Kiểu nhà": "Loại nhà (ví dụ: nhà phố, biệt thự, căn hộ, ...).",
    "R2": "R-squared, hệ số xác định, đo lường mức độ phù hợp của mô hình.",
    "MSE": "Mean Squared Error, sai số bình phương trung bình, đo lường độ chính xác của mô hình.",
    "MAE": "Mean Absolute Error, sai số tuyệt đối trung bình, đo lường độ chính xác của mô hình."
}

def clear_frame():
    for widget in root.winfo_children():
        widget.destroy()

def main_menu():
    clear_frame()
    root.minsize(height=800, width=800)

    # Tạo khung chính
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Khung menu phía trên cùng
    menu_frame = tk.Frame(main_frame, bg='lightgray')
    menu_frame.pack(side=tk.TOP, fill=tk.X)

    # Tạo các nút menu và đặt chúng bên phải
    button_frame = tk.Frame(menu_frame, bg='lightgray')
    button_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    tk.Button(button_frame, text="Mua nhà", width=10, command=menu_mua_nha).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Thuê nhà", width=10, command=menu_thue_nha).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Giá nhà đất", width=10, command=menu_gia_dat).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Dự đoán", width=10, command=show_prediction_menu).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Đăng nhập", width=10, command=show_login_frame).pack(side=tk.LEFT, padx=5)

    # Thanh tìm kiếm
    search_frame = tk.Frame(main_frame)
    search_frame.pack(pady=30)

    tk.Label(search_frame, text="Tìm kiếm:").pack(side=tk.LEFT, padx=5)
    search_entry = tk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, padx=5)

    tk.Button(search_frame, text="Tìm", command=lambda: search_house(search_entry.get())).pack(side=tk.LEFT, padx=5)

    # Lọc dữ liệu
    filter_frame = tk.Frame(main_frame)
    filter_frame.pack(pady=10)

    tk.Label(filter_frame, text="Khu vực:").pack(side=tk.LEFT, padx=5)
    area_combobox = ttk.Combobox(filter_frame, values=["Toàn quốc", "Hà Nội", "TP Hồ Chí Minh", "Đà Nẵng", "Thanh Hóa", "Nghệ An"])
    area_combobox.pack(side=tk.LEFT, padx=5)

    tk.Label(filter_frame, text="Loại bất động sản:").pack(side=tk.LEFT, padx=5)
    property_type_combobox = ttk.Combobox(filter_frame, values=["Chung cư", "Nhà phố", "Biệt thự"])
    property_type_combobox.pack(side=tk.LEFT, padx=5)

    tk.Label(filter_frame, text="Giá bán:").pack(side=tk.LEFT, padx=5)
    price_combobox = ttk.Combobox(filter_frame, values=["1-2 tỷ", "2-5 tỷ", "5-10 tỷ", "Trên 10 tỷ"])
    price_combobox.pack(side=tk.LEFT, padx=5)

    tk.Button(filter_frame, text="Lọc", command=lambda: filter_house(area_combobox.get(), property_type_combobox.get(), price_combobox.get())).pack(side=tk.LEFT, padx=5)

    # Bảng hiển thị danh sách nhà
    tree = ttk.Treeview(main_frame, columns=("name", "address", "price", "type"), show="headings")
    tree.heading("name", text="Tên nhà")
    tree.heading("address", text="Địa chỉ")
    tree.heading("price", text="Giá")
    tree.heading("type", text="Loại nhà")
    tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

def search_house(query):
    print(f"Tìm kiếm với từ khóa: {query}")
    # Thêm logic tìm kiếm vào đây

def filter_house(area, property_type, price):
    print(f"Filtrer với khu vực: {area}, Loại: {property_type}, Giá: {price}")
    # Thêm logic lọc vào đây


def show_login_frame():
    clear_frame()
    root.minsize(height=800, width=800)
    login_frame = LoginFrame(root, main_menu)
    login_frame.pack(fill=tk.BOTH, expand=True)

class BaseFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

class LoginFrame(BaseFrame):
    def __init__(self, parent, return_callback):
        BaseFrame.__init__(self, parent)
        self.return_callback = return_callback

        label = tk.Label(self, text="Đăng nhập", font=("Helvetica", 20))
        label.pack(pady=10)

        tk.Label(self, text="Tên đăng nhập:").pack(pady=5)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack(pady=5)

        tk.Label(self, text="Mật khẩu:").pack(pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack(pady=5)

        login_button = tk.Button(self, text="Đăng nhập", command=self.login)
        login_button.pack(pady=10)

        back_button = tk.Button(self, text="Quay lại", command=self.return_callback)
        back_button.pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        # Thêm logic đăng nhập vào đây
        print(f"Tên đăng nhập: {username}, Mật khẩu: {password}")

def show_prediction_menu():

    clear_frame()
    root.minsize(height=800, width=800)

    # Tạo khung chính
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Khung menu phía trái
    menu_frame = tk.Frame(main_frame, bg='lightgray')
    menu_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Tạo các nút menu cho dự đoán và đặt chúng từ trên xuống dưới
    prediction_button_frame = tk.Frame(menu_frame, bg='lightgray')
    prediction_button_frame.pack(side=tk.TOP, padx=10, pady=10)

    tk.Button(prediction_button_frame, text="Nhà ở", width=10, command=menu_du_doan_nha_o).pack(pady=5)
    tk.Button(prediction_button_frame, text="Chung cư", width=10, command=menu_du_doan_chung_cu).pack(pady=5)
    tk.Button(prediction_button_frame, text="Quay lại", width=10, command=main_menu).pack(pady=5)


# hàm hiển thị từ điển giải thích
def show_explanation():
    explanation_window = tk.Toplevel(root)
    explanation_window.title("Giải thích các thuật ngữ")
    explanation_window.geometry("400x300")

    explanation_text = tk.Text(explanation_window, wrap=tk.WORD)
    explanation_text.pack(expand=True, fill=tk.BOTH)

    for term, explanation in terms_explanation.items():
        explanation_text.insert(tk.END, "{}: {}\n\n".format(term, explanation))

# hàm tìm các ngôi nhà tương đồng
def nha_tuongTu():

    # tao dataFarme
    row = {}
    row["Diện tích"] = float(entry_dienTich.get())
    row["Số tầng"] = int(entry_soTang.get())
    row["Số phòng ngủ"] = int(entry_soPhongNgu.get())
    row["WC"] = int(entry_wc.get())
    row["Thang Máy"] = int(entry_thangMay.get())
    row["Mặt tiền"] = float(entry_matTien.get())
    row["kieunha"] = 'Nhà'

    str1 = district_quan.get()
    row["Quan"] = str1
    str2 = district_vitri.get()
    row["Vitri"] = str2
    str3 = district_phaply.get()
    row["phaply"] = str3

    input_data = pd.DataFrame([row])

    nhaTuongTu = find_similar_houses(input_data)
    return nhaTuongTu


def chungCu_tungTu():

    # tao dataFarme
    row = {}
    row["Diện tích"] = float(entry_dienTich_chungCu.get())
    row["Số tầng"] = int(1)
    row["Số phòng ngủ"] = int(entry_soPhongNgu_chungCu.get())
    row["WC"] = int(entry_wc_chungCu.get())
    row["Thang Máy"] = int(0)
    row["Mặt tiền"] = float(0)
    row["kieunha"] = 'Chung cư'

    str1 = district_quan_chungCu.get()
    row["Quan"] = str1
    row["Vitri"] = None
    str3 = district_phaply_chungCu.get()
    row["phaply"] = str3

    input_data = pd.DataFrame([row])

    chungCuTuongTu = find_similar_chungCu(input_data)
    return chungCuTuongTu

def star_nha():

    # tao dataFarme
    row = {}
    row["Diện tích"] = float(entry_dienTich.get())
    row["Số tầng"] = int(entry_soTang.get())
    row["Số phòng ngủ"] = int(entry_soPhongNgu.get())
    row["WC"] = int(entry_wc.get())
    row["Thang Máy"] = int(entry_thangMay.get())
    row["Mặt tiền"] = float(entry_matTien.get())
    row["kieunha"] = int(2)

    str1 = district_quan.get()
    if str1 in quan_values:
        row["Quan"] = quan_values[str1]
    str2 = district_vitri.get()
    if str2 in vitri_values:
        row["Vitri"] = vitri_values[str2]
    str3 = district_phaply.get()
    if str3 in phaply_values:
        row["phaply"] = phaply_values[str3]

    input_data = pd.DataFrame([row])

    # Chạy thuật toán và lấy thời gian chạy
    giaNha, r2, mse, mae = predict_price(input_data)
    giaNha = giaNha[0]
    predicted_price.config(text="Giá: {:.3f} tỷ".format(giaNha))
    predicted_R2.config(text="R2: {:.3f}" .format(r2))
    predicted_MSE.config(text="MSE: {:.3f}".format(mse))
    predicted_MAE.config(text="MAE: {:.3f}".format(mae))

    results = nha_tuongTu()
    filtered_results = results[abs(results["Giá nhà(Tỷ)"] - giaNha) <= 1.5]

    # Hiển thị DataFrame dưới dạng bảng trong Tkinter
    for row in tree.get_children():
        tree.delete(row)
    # Thêm dữ liệu từ DataFrame
    for index, row in filtered_results.iterrows():
        tree.insert("", "end", values=list(row))


def star_chungCu():

    # tao dataFarme
    row = {}
    row["Diện tích"] = float(entry_dienTich_chungCu.get())
    row["Số tầng"] = int(1)
    row["Số phòng ngủ"] = int(entry_soPhongNgu_chungCu.get())
    row["WC"] = int(entry_wc_chungCu.get())
    row["Thang Máy"] = int(0)
    row["Mặt tiền"] = float(0)
    row["kieunha"] = int(0)

    str1 = district_quan_chungCu.get()
    if str1 in quan_values:
        row["Quan"] = quan_values[str1]

    str3 = district_phaply_chungCu.get()
    if str3 in phaply_values:
        row["phaply"] = phaply_values[str3]

    row["Vitri"] = int(1)
    input_data = pd.DataFrame([row])

    # Chạy thuật toán và lấy thời gian chạy
    giaNha, r2, mse, mae = predict_price(input_data)
    giaNha = giaNha[0] - 2.5
    predicted_price_chungCu.config(text="Giá: {:.3f} tỷ".format(giaNha))
    predicted_R2_chungCu.config(text="R2: {:.3f}" .format(r2))
    predicted_MSE_chungCu.config(text="MSE: {:.3f}".format(mse))
    predicted_MAE_chungCu.config(text="MAE: {:.3f}".format(mae))

    results = chungCu_tungTu()
    filtered_results = results[abs(results["Giá nhà(Tỷ)"] - giaNha) <= 4]
    # Chọn các cột cần hiển thị
    filtered_results = filtered_results[["Quận", "Diện tích", "Số phòng ngủ", "WC", "Kiểu nhà", "Pháp lý", "Giá nhà(Tỷ)"]]

    # Hiển thị DataFrame dưới dạng bảng trong Tkinter
    for row in tree_chungCu.get_children():
        tree_chungCu.delete(row)
    # Thêm dữ liệu từ DataFrame
    for index, row in filtered_results.iterrows():
        tree_chungCu.insert("", "end", values=list(row))

def menu_du_doan_nha_o():

    clear_frame()

    show_prediction_menu()

    root.minsize(height=800, width=800)
    # Tạo tiêu đề
    title_label = tk.Label(root, text="Nhập thông tin căn nhà ", font=("Arial", 10))
    title_label.place(x=150, y=0)

    # Tạo menu thả xuống để chọn quận
    global district_quan
    district_label = tk.Label(root, text="Chọn quận:")
    district_label.place(x=150, y=25)

    district_quan = ttk.Combobox(root, values=districts)
    district_quan.place(x=150, y=50)

    # Tạo menu thả xuống để chọn Vị trí
    global district_vitri
    district_label = tk.Label(root, text="Vị trí căn nhà:")
    district_label.place(x=150, y=80)

    district_vitri = ttk.Combobox(root, values=viTri)
    district_vitri.place(x=150, y=105)

    # Tạo menu thả xuống để chọn pháp lý
    global district_phaply
    district_label = tk.Label(root, text="Giấy tờ:")
    district_label.place(x=150, y=135)

    district_phaply = ttk.Combobox(root, values=phapLy)
    district_phaply.place(x=150, y=160)

    # Diện tích
    lbl_dienTich = tk.Label(root, text="Diện tích:")
    lbl_dienTich.place(x=400, y=25)
    global entry_dienTich
    entry_dienTich = tk.Entry(root, width=15)
    entry_dienTich.place(x=400, y=50)

    # Số tầng
    lbl_soTang = tk.Label(root, text="Số tầng:")
    lbl_soTang.place(x=630, y=25)
    global entry_soTang
    entry_soTang = tk.Entry(root, width=15)
    entry_soTang.place(x=630, y=50)

    # Số phòng ngủ
    lbl_soPhongNgu = tk.Label(root, text="Số phòng ngủ:")
    lbl_soPhongNgu.place(x=400, y=80)
    global entry_soPhongNgu
    entry_soPhongNgu = tk.Entry(root, width=15)
    entry_soPhongNgu.place(x=400, y=105)

    # WC
    lbl_wc = tk.Label(root, text="Số phòng WC:")
    lbl_wc.place(x=630, y=80)
    global entry_wc
    entry_wc = tk.Entry(root, width=15)
    entry_wc.place(x=630, y=105)

    # Thang máy
    lbl_thangMay = tk.Label(root, text="Số thang máy:")
    lbl_thangMay.place(x=400, y=135)
    global entry_thangMay
    entry_thangMay = tk.Entry(root, width=15)
    entry_thangMay.place(x=400, y=160)

    # Mặt tiền
    lbl_matTien = tk.Label(root, text="Mặt tiền:")
    lbl_matTien.place(x=630, y=135)
    global entry_matTien
    entry_matTien = tk.Entry(root, width=15)
    entry_matTien.place(x=630, y=160)

    # Tạo nút dự đoán
    predict_button = tk.Button(root, text="Dự đoán", command=star_nha)
    predict_button.place(x=150, y=190)

    # hiển thị giá căn nhà
    global predicted_price
    predicted_price = tk.Label(root, text="Giá: ")
    predicted_price.place(x=150, y=220)


    # R2
    global predicted_R2
    predicted_R2 = tk.Label(root, text="R2: ")
    predicted_R2.place(x=150, y=245)

    # MSE
    global predicted_MSE
    predicted_MSE = tk.Label(root, text="MSE: ")
    predicted_MSE.place(x=150, y=270)

    # MAE
    global predicted_MAE
    predicted_MAE = tk.Label(root, text="MAE: ")
    predicted_MAE.place(x=150, y=295)

    # Tạo nút giải thích các thuật ngữ
    explain_button = tk.Button(root, text="Giải thích các thuật ngữ", command=show_explanation)
    explain_button.place(x=150, y=385)

    # Frame để chứa Treeview và thanh cuộn
    frame = tk.Frame(root)
    frame.place(x=350, y=220, width=400, height=200)

    # Hiển thị Treeview
    global tree
    columns = (
    "Quận", "Diện tích", "Số tầng", "Số phòng ngủ", "WC", "Thang Máy", "Mặt tiền", "Vị trí", "Kiểu nhà", "Pháp lý",
    "Giá nhà(Tỷ)")
    tree = ttk.Treeview(frame, columns=columns, show="headings")

    # Đặt tên và độ rộng cột
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)  # Đặt độ rộng cột để chỉ hiển thị tối đa 4 cột

    # Thêm thanh cuộn
    scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar_y.pack(side="right", fill="y")

    scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    scrollbar_x.pack(side="bottom", fill="x")

    tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    tree.pack(fill='both', expand=True)


def menu_du_doan_chung_cu():

    clear_frame()

    show_prediction_menu()
    root.minsize(height=800, width=800)
    # Tạo tiêu đề
    title_label = tk.Label(root, text="Nhập thông tin căn nhà ", font=("Arial", 10))
    title_label.place(x=150, y=0)

    # Tạo menu thả xuống để chọn quận
    global district_quan_chungCu
    district_label = tk.Label(root, text="Chọn quận:")
    district_label.place(x=150, y=25)

    district_quan_chungCu = ttk.Combobox(root, values=districts)
    district_quan_chungCu.place(x=150, y=50)


    # Tạo menu thả xuống để chọn pháp lý
    global district_phaply_chungCu
    district_label = tk.Label(root, text="Giấy tờ:")
    district_label.place(x=150, y=80)

    district_phaply_chungCu = ttk.Combobox(root, values=phapLy)
    district_phaply_chungCu.place(x=150, y=105)

    # Diện tích
    lbl_dienTich = tk.Label(root, text="Diện tích:")
    lbl_dienTich.place(x=400, y=25)
    global entry_dienTich_chungCu
    entry_dienTich_chungCu = tk.Entry(root, width=15)
    entry_dienTich_chungCu.place(x=400, y=50)


    # Số phòng ngủ
    lbl_soPhongNgu = tk.Label(root, text="Số phòng ngủ:")
    lbl_soPhongNgu.place(x=400, y=80)
    global entry_soPhongNgu_chungCu
    entry_soPhongNgu_chungCu = tk.Entry(root, width=15)
    entry_soPhongNgu_chungCu.place(x=400, y=105)

    # WC
    lbl_wc = tk.Label(root, text="Số phòng WC:")
    lbl_wc.place(x=630, y=25)
    global entry_wc_chungCu
    entry_wc_chungCu = tk.Entry(root, width=15)
    entry_wc_chungCu.place(x=630, y=50)


    # Tạo nút dự đoán
    predict_button = tk.Button(root, text="Dự đoán", command=star_chungCu)
    predict_button.place(x=150, y=190)


    # hiển thị giá căn chung cu
    global predicted_price_chungCu
    predicted_price_chungCu = tk.Label(root, text="Giá: ")
    predicted_price_chungCu.place(x=150, y=220)


    # R2
    global predicted_R2_chungCu
    predicted_R2_chungCu = tk.Label(root, text="R2: ")
    predicted_R2_chungCu.place(x=150, y=245)

    # MSE
    global predicted_MSE_chungCu
    predicted_MSE_chungCu = tk.Label(root, text="MSE: ")
    predicted_MSE_chungCu.place(x=150, y=270)

    # MAE
    global predicted_MAE_chungCu
    predicted_MAE_chungCu = tk.Label(root, text="MAE: ")
    predicted_MAE_chungCu.place(x=150, y=295)

    # Tạo nút giải thích các thuật ngữ
    explain_button = tk.Button(root, text="Giải thích các thuật ngữ", command=show_explanation)
    explain_button.place(x=150, y=385)

    # Frame để chứa Treeview và thanh cuộn
    frame = tk.Frame(root)
    frame.place(x=350, y=220, width=400, height=200)

    # Hiển thị Treeview
    global tree_chungCu
    columns = ("Quận", "Diện tích", "Số phòng ngủ", "WC", "Kiểu nhà", "Pháp lý", "Giá nhà(Tỷ)")
    tree_chungCu = ttk.Treeview(frame, columns=columns, show="headings")

    # Đặt tên và độ rộng cột
    for col in columns:
        tree_chungCu.heading(col, text=col)
        tree_chungCu.column(col, anchor="center", width=100)  # Đặt độ rộng cột để chỉ hiển thị tối đa 4 cột

    # Thêm thanh cuộn
    scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree_chungCu.yview)
    scrollbar_y.pack(side="right", fill="y")

    scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree_chungCu.xview)
    scrollbar_x.pack(side="bottom", fill="x")

    tree_chungCu.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    tree_chungCu.pack(fill='both', expand=True)


def menu_mua_nha():
    print("Chức năng Mua nhà được chọn.")

def menu_thue_nha():
    print("Chức năng Thuê nhà được chọn.")

def menu_gia_dat():
    print("Chức năng Giá nhà đất được chọn.")

# Khởi tạo giao diện tkinter
root = tk.Tk()
root.title("Ứng dụng quản lý nhà đất")

# Hiển thị menu chính khi chạy chương trình
main_menu()

# Chạy vòng lặp main của tkinter
root.mainloop()
