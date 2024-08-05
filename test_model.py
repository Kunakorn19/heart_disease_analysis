# train_model.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# โหลดข้อมูล (ปรับให้ตรงกับข้อมูลจริงของคุณ)
# ตัวอย่างนี้ใช้ข้อมูลจำลอง
# df = pd.read_csv('your_data.csv')  # ปรับให้ตรงกับไฟล์ข้อมูลของคุณ
# X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach']]  # ฟีเจอร์
# y = df['target']  # ตัวแปรเป้าหมาย

# ข้อมูลตัวอย่าง
data = {
    'age': [63, 37, 41, 56, 57],
    'sex': [1, 1, 0, 1, 0],
    'cp': [3, 2, 1, 1, 2],
    'trestbps': [145, 130, 120, 140, 130],
    'chol': [233, 250, 204, 236, 245],
    'fbs': [1, 1, 0, 0, 1],
    'thalach': [150, 187, 172, 178, 165],
    'target': [1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach']]
y = df['target']

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทดสอบโมเดล
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# บันทึกโมเดล
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

