import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# อ่านข้อมูลจากไฟล์ CSV
file_path = '1heart.csv'  # ใช้ชื่อไฟล์ที่ถูกต้อง
try:
    data = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# การจัดการข้อมูลที่ขาดหายไป
data = data.dropna()

# การแบ่งข้อมูลเป็นชุดฝึกอบรมและชุดทดสอบ
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกอบรมโมเดล Machine Learning
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ข้อมูลของบุคคลตัวอย่าง (ตัวอย่าง: [age, sex, cp, trestbps, chol, fbs, thalach])
sample_data = pd.DataFrame({
    'age': [63],
    'sex': [1],
    'cp': [0],
    'trestbps': [80],
    'chol': [0],
    'fbs': [100],
    'thalach': [100]
})

# ทำนายผลลัพธ์
prediction = model.predict(sample_data)

# แสดงผลลัพธ์
if prediction[0] == 1:
    print("คุณมีความเสี่ยงเป็นโรคหัวใจ")
else:
    print("คุณไม่มีความเสี่ยงเป็นโรคหัวใจ")
    
