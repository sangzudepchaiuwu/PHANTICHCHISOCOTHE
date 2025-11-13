import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# ====== 1️ Đọc dữ liệu ======
df = pd.read_csv(r"D:\chuyendoiso\Data\data.csv")

# ====== 2️ Tính BMI ======
# Cột 'bmi' vẫn cần được tính để GÁN NHÃN ở bước tiếp theo.
df['bmi'] = df['can_nang_kg'] / ((df['chieu_cao_cm']/100) ** 2)

# ====== 3️ Gán nhãn thể trạng ======
def classify_bmi(bmi):
    if bmi < 16:
        return 0    # Thiếu cân nghiêm trọng
    elif bmi < 18.5:
        return 1    # Thiếu cân
    elif bmi < 25:
        return 2    # Bình thường
    elif bmi < 30:
        return 3    # Thừa cân
    else:
        return 4    # Béo phì

# Cột 'label' (nhãn) vẫn được gán từ 'bmi'
df['label'] = df['bmi'].apply(classify_bmi)

# --- THAY ĐỔI QUAN TRỌNG NHẤT TẠI ĐÂY ---
# ====== 4️ Chọn đặc trưng (X) và nhãn (y) ======
X = df[['chieu_cao_cm','can_nang_kg','calo_nap','calo_tieu_hao','thoi_gian_ngu']]
y = df['label']
# -----------------------------------------------

# ====== 5️ Tách train/test ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 6️ Cân bằng dữ liệu bằng SMOTE ======
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ====== 7️ Huấn luyện Random Forest ======
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train_res, y_train_res)

# ====== 8️ Đánh giá mô hình ======
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(" Độ chính xác tổng thể:", round(acc, 4))
print("\n Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ====== 9️ Trực quan hóa ======

# --- 9a: Ma trận nhầm lẫn ---
plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Thiếu cân nghiêm trọng", "Thiếu cân", "Bình thường", "Thừa cân", "Béo phì"]
)
disp.plot(cmap='Blues', values_format='d')
plt.title("Ma trận nhầm lẫn - Random Forest (5 lớp) - Predict by Attributes")
plt.show()

# --- 9b: Phân bố BMI theo nhóm thể trạng ---
plt.figure(figsize=(8,5))
for label, group in df.groupby('label'):
    plt.hist(group['bmi'], bins=20, alpha=0.6, label=f'Nhóm {label}')
plt.title("Phân bố BMI theo từng thể trạng")
plt.xlabel("BMI")
plt.ylabel("Tần suất")
plt.legend()
plt.show()

# ====== 10 Lưu mô hình ======
model_path = r"D:\chuyendoiso\model_classes.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"\n Mô hình 5 lớp đã được lưu tại: {model_path}")