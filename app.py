import sys  
import warnings
import cv2
import pickle
import numpy as np
import sklearn

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from cancer import Ui_MainWindow
from trieuchung import Ui_MainWindow as TrieuChungUI


from tensorflow.keras.models import load_model

model_image = load_model("model_test4.h5", compile=False)


with open("model.pkl", "rb") as file:
    model = pickle.load(file)

model_features = [col.strip() for col in model.feature_names_in_]
#print("Tên cột của mô hình sau khi chuẩn hóa:", model_features)

symptom_translation = {
    "GENDER": "Giới tính",
    "AGE": "Tuổi",
    "SMOKING": "Hút thuốc",
    "YELLOW_FINGERS": "Vàng ngón tay",
    "ANXIETY": "Lo lắng",
    "PEER_PRESSURE": "Áp lực",
    "CHRONIC DISEASE": "Bệnh mãn tính",
    "FATIGUE": "Mệt mỏi",
    "ALLERGY": "Dị ứng",
    "WHEEZING": "Thở khò khè",
    "ALCOHOL CONSUMING": "Uống rượu",
    "COUGHING": "Ho",
    "SHORTNESS OF BREATH": "Hụt hơi",
    "SWALLOWING DIFFICULTY": "Khó nuốt",
    "CHEST PAIN": "Đau ngực"
}

def map_gender(gender):
    return 0 if gender == 'M' else 1


class TrieuChungWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = TrieuChungUI()
        self.ui.setupUi(self)
        self.ui.reset_button.clicked.connect(self.reset_symptoms)

        self.setup_symptom_table()

     
        self.ui.predict_button.clicked.connect(self.predict_symptoms)

   
    def setup_symptom_table(self):
        
        self.ui.symptom_table.setColumnCount(3)
        self.ui.symptom_table.setRowCount(len(model_features) - 2) 
        self.ui.symptom_table.setHorizontalHeaderLabels(["Triệu chứng", "Có", "Không"])
        self.ui.symptom_table.setColumnWidth(0, 250)
        self.ui.symptom_table.setColumnWidth(1, 150)
        self.ui.symptom_table.setColumnWidth(2, 150)

        for i, symptom in enumerate(model_features[2:]):
            symptom_vietnamese = symptom_translation.get(symptom, symptom) 
            self.ui.symptom_table.setItem(i, 0, QTableWidgetItem(symptom_vietnamese))

            checkbox_yes = QTableWidgetItem()
            checkbox_yes.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_yes.setCheckState(Qt.Unchecked)
            self.ui.symptom_table.setItem(i, 1, checkbox_yes)
    
            checkbox_no = QTableWidgetItem()
            checkbox_no.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_no.setCheckState(Qt.Unchecked)
            self.ui.symptom_table.setItem(i, 2, checkbox_no)

    def predict_symptoms(self):
        model_features = [col.strip() for col in model.feature_names_in_]

        try:
            age = self.ui.spinBox_2.value()
            if age <= 0:
                QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tuổi hợp lệ.")
                return

            gender = self.ui.Tuoi_3.currentText()
            gender_value = map_gender(gender)
        except AttributeError:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy widget nhập Tuổi/Giới tính. Vui lòng kiểm tra lại.")
            return

        # Mặc định: tất cả triệu chứng là 1 (tức là "Không")
        input_data = {col: 1 for col in model_features}
        input_data["GENDER"] = gender_value
        input_data["AGE"] = age

        missing_symptoms = []

        for row in range(self.ui.symptom_table.rowCount()):
            yes_item = self.ui.symptom_table.item(row, 1)
            no_item = self.ui.symptom_table.item(row, 2)
            symptom_vietnamese = self.ui.symptom_table.item(row, 0).text().strip()
            symptom_english = {v.strip(): k.strip() for k, v in symptom_translation.items()}.get(symptom_vietnamese,
                                                                                                 None)

            if symptom_english and yes_item and no_item:
                yes_checked = yes_item.checkState() == Qt.Checked
                no_checked = no_item.checkState() == Qt.Checked

                if yes_checked and no_checked:
                    QMessageBox.warning(self, "Lỗi",
                                        f"Không thể chọn cả 'Có' và 'Không' cho triệu chứng: {symptom_vietnamese}")
                    return
                elif yes_checked:
                    input_data[symptom_english] = 2
                elif no_checked:
                    input_data[symptom_english] = 1
                else:
                    missing_symptoms.append(symptom_vietnamese)

        # Nếu có triệu chứng bị bỏ trống
        if missing_symptoms:
            missing_text = "\n".join(missing_symptoms)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thiếu dữ liệu")
            msg.setText("Một số triệu chứng chưa được chọn:")
            msg.setInformativeText(f"{missing_text}\n\nBạn có muốn bỏ qua và mặc định là 'Không'?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = msg.exec_()

            if result == QMessageBox.No:
                return  # Dừng lại để người dùng quay lại chọn tiếp

            # Nếu chọn Yes: giữ nguyên mặc định = 1 (đã set ở input_data)

        # Đảm bảo đúng thứ tự cột đầu vào
        input_vector = pd.DataFrame([input_data], columns=model.feature_names_in_)

        # Dự đoán bằng mô hình
        prediction = model.predict(input_vector)
        result_mapping = {0: "Bình thường", 1: "Nguy cơ mắc ung thư phổi"}
        result_text = result_mapping.get(prediction[0], "Không xác định")

        self.ui.result_lineEdit.setText(result_text)
        QMessageBox.information(self, "Kết quả dự đoán", result_text)


    def reset_symptoms(self):
        # Reset tất cả checkbox trong bảng triệu chứng
        for row in range(self.ui.symptom_table.rowCount()):
            for col in [1, 2]:
                item = self.ui.symptom_table.item(row, col)
                if item:
                    item.setCheckState(Qt.Unchecked)


        self.ui.spinBox_2.setValue(0)
        self.ui.Tuoi_3.setCurrentIndex(0)

        self.ui.result_lineEdit.clear()


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.input_image_button.clicked.connect(self.load_image)
        self.ui.processing_button.clicked.connect(self.process_image)
        self.ui.reset_button.clicked.connect(self.reset_app)
        self.ui.exit_button.clicked.connect(self.close)
        self.ui.symptom_button.clicked.connect(self.open_symptom_window)

        self.image_path = ""

    def load_image(self):
       
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.ui.input_image_label.setPixmap(pixmap.scaled(350, 350))

    def process_image(self):
        if self.image_path:
            # Đọc ảnh bất kỳ
            image = cv2.imread(self.image_path)
            # image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc ảnh. Vui lòng kiểm tra định dạng hoặc đường dẫn.")
                return

            # Nếu ảnh là màu (3 kênh), chuyển sang xám
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize ảnh về đúng kích thước mô hình cần
            image = cv2.resize(image, (150, 150))

            # Chuẩn hóa giá trị pixel về khoảng [0, 1]
            image = image / 255.0

            # Mở rộng chiều để khớp với input model (1 ảnh, 1 kênh)
            image = np.expand_dims(image, axis=-1)  # (150,150) → (150,150,1)
            image = np.expand_dims(image, axis=0)  # (150,150,1) → (1,150,150,1)

            #Dự đoán bằng mô hình
            prediction = model_image.predict(image)
            label = 1 if prediction[0][0] > 0.5 else 0

            #Hiển thị kết quả
            result = "Có dấu hiệu ung thư phổi" if label == 1 else "Bình thường"
            self.ui.result_label.setText(result)
            QMessageBox.information(self, "Kết quả phân loại", result)

        else:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một ảnh trước khi xử lý.")

    # def classify_image(self):
    #     """Phân loại ảnh bằng quy tắc đơn giản"""
    #     if self.image_path:
    #         image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
    #         brightness = np.mean(image)
    #
    #         if brightness < 100:
    #             result = "Có dấu hiệu ung thư phổi"
    #         else:
    #             result = "Không có dấu hiệu ung thư phổi"
    #
    #         self.ui.result_label.setText(result)
    #         QMessageBox.information(self, "Kết quả phân loại", result)

    def reset_app(self):
        
        self.ui.input_image_label.clear()
        self.ui.result_label.clear()
        self.image_path = ""



    def open_symptom_window(self):
       
        self.symptom_window = TrieuChungWindow()
        self.symptom_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
