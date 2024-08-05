import streamlit as st
import pickle
import numpy as np

# โหลดโมเดลที่บันทึกไว้
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ตั้งชื่อแอป
st.title("ระบบวิเคราะห์ความเสี่ยงต่อการโรคหัวใจ")

# รับข้อมูลจากผู้ใช้
age = st.number_input("อายุ", min_value=0, max_value=120, value=0)
sex = st.selectbox("เพศ", ["ชาย", "หญิง"])
cp = st.selectbox("ระดับความเจ็บหน้าอก", [0, 1, 2, 3])
trestbps = st.number_input("ความดันโลหิตขณะพัก", min_value=0, max_value=200, value=120)
chol = st.number_input("คอเลสเตอรอล", min_value=0, max_value=600, value=200)
fbs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร มีค่ามากกว่า 120", ["จริง", "เท็จ"])
thalach = st.number_input("อัตราการเต้นของหัวใจสูงสุด", min_value=0, max_value=220, value=150)

# แปลงข้อมูลที่ได้รับให้เป็นรูปแบบที่โมเดลต้องการ
sex = 1 if sex == "ชาย" else 0
fbs = 1 if fbs == "จริง" else 0

# สร้าง array ของข้อมูล
data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach]])

# ปุ่มสำหรับทำนาย
if st.button("ตรวจวิเคราะห์"):
    # ทำการคาดการณ์
    prediction = model.predict(data)
    result = 'เสี่ยง' if prediction[0] == 1 else 'ไม่เสี่ยง'
    
    # แสดงผลลัพธ์
    if prediction[0] == 1:
        st.markdown(
            "<h1 style='text-align: center; color: red;'>คุณมีความเสี่ยงต่อการเป็นโรคหัวใจ</h1>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h1 style='text-align: center; color: green;'>คุณไม่เสี่ยงต่อการเป็นโรคหัวใจ</h1>",
            unsafe_allow_html=True
        )
