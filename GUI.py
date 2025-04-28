import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler

# Load models
try:
    ann_model = load_model('models/heart_disease_model_new.h5')
    naive_bayes_model = joblib.load('models/naive_bayes_model.pkl')
    decision_tree_model = joblib.load('models/decision_tree_model.pkl')
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình hoặc scaler: {e}")

# Giao diện chính
st.title('❤️ Dự đoán nguy cơ mắc bệnh tim')
st.write("Hãy nhập thông tin sức khỏe của bạn để hệ thống dự đoán nguy cơ bệnh tim tiềm ẩn.")

# --- CHỌN MÔ HÌNH ---
st.sidebar.header("⚙️ Lựa chọn mô hình dự đoán")
model_choice = st.sidebar.selectbox(
    "Chọn mô hình muốn sử dụng",
    ["ANN", "Naive Bayes", "Decision Tree", "Combine"]
)

with st.form("heart_disease_form"):
    st.header("Thông tin sức khỏe cá nhân 🩺")

    col1, col2 = st.columns(2)

    with col1:
        highBP = st.radio('Bạn có bị huyết áp cao?', ['Không', 'Có'])
        highChol = st.radio('Bạn có cholesterol cao?', ['Không', 'Có'])
        smoker = st.radio('Bạn có hút thuốc?', ['Không', 'Có'])
        stroke = st.radio('Bạn từng bị đột quỵ?', ['Không', 'Có'])
        diabetes = st.radio('Bạn đang bị tiểu đường?', ['Chưa bị', 'Nguy cơ', 'Đã mắc'])
        hvy = st.radio('Bạn có sử dụng bia rượu thường xuyên?', ['Không', 'Có'])

    with col2:
        physActivity = st.radio('Bạn có tập luyện thể thao?', ['Không', 'Có'])
        diffWalk = st.radio('Bạn có khó khăn khi đi lại?', ['Không', 'Có'])
        bmi = st.number_input('Chỉ số BMI (kg/m²)', 0, 100, 25)
        genHealth = st.selectbox('Sức khỏe tổng quát (1: Tốt - 5: Kém)', [1, 2, 3, 4, 5])
        mentHealth = st.number_input('Số ngày tinh thần kém trong 30 ngày', 0, 30, 0)
        physHealth = st.number_input('Số ngày thể chất kém trong 30 ngày', 0, 30, 0)

    st.header("Thông tin cá nhân 👤")
    sex = st.selectbox('Giới tính', ['Nam', 'Nữ', 'Không tiện nói ra'])
    age = st.number_input('Tuổi của bạn', 1, 120, 25)

    submitted = st.form_submit_button("🔍 Dự đoán bệnh tim")

if submitted:
    with st.spinner('Đang phân tích dữ liệu...'):
        # Chuẩn bị dữ liệu đầu vào
        input_data = {
        }
        if model_choice != "ANN":
            input_data = {
                'HighBP': 1 if highBP == 'Có' else 0,
                'HighChol': 1 if highChol == 'Có' else 0,
                'BMI': bmi,
                'Smoker': 1 if smoker == 'Có' else 0,
                'Stroke': 1 if stroke == 'Có' else 0,
                'Diabetes': 1 if diabetes == 'Nguy cơ' else 2 if diabetes == 'Đã mắc' else 0,
                'PhysActivity': 1 if physActivity == 'Có' else 0,
                'HvyAlcoholConsump': 1 if hvy == 'Có' else 0,
                'GenHlth': genHealth,
                'MentHlth': mentHealth,
                'PhysHlth': physHealth,
                'DiffWalk': 1 if diffWalk == 'Có' else 0,
                'Sex': 1 if sex == 'Nam' else 0,
                'Age': age / 5
            }
        else:
            input_data = {
            'HighBP': 1 if highBP == 'Có' else 0,
            'HighChol': 1 if highChol == 'Có' else 0,
            'BMI': bmi,
            'Smoker': 1 if smoker == 'Có' else 0,
            'Stroke': 1 if stroke == 'Có' else 0,
            'Diabetes': 1 if diabetes == 'Nguy cơ' else 2 if diabetes == 'Đã mắc' else 0,
            'PhysActivity': 1 if physActivity == 'Có' else 0,
            'GenHlth': genHealth,
            'MentHlth': mentHealth,
            'PhysHlth': physHealth,
            'DiffWalk': 1 if diffWalk == 'Có' else 0,
            'Sex': 1 if sex == 'Nam' else 0,
            'Age': age / 5
        }
        input_df = pd.DataFrame([input_data])

        
        # Dự đoán
        if model_choice == "ANN":
            scaler = joblib.load('scaler/scaler.pkl')
            input_df_scaled = scaler.transform(input_df)
            prediction = ann_model.predict(input_df_scaled)
            probability = prediction[0][0]
        elif model_choice == "Naive Bayes":
            print(naive_bayes_model.predict(input_df))
            probability = naive_bayes_model.predict_proba(input_df)[0][1]
        elif model_choice == "Decision Tree":
            probability = decision_tree_model.predict_proba(input_df)[0][1]
        else:
            probability2 = naive_bayes_model.predict_proba(input_df)[0][1]
            probability3 = decision_tree_model.predict_proba(input_df)[0][1]
            if 'HvyAlcoholConsump' in input_data:
                del input_data['HvyAlcoholConsump']
            input_df = pd.DataFrame([input_data])
            scaler = joblib.load('scaler/scaler.pkl')
            input_df_scaled = scaler.transform(input_df)
            prediction = ann_model.predict(input_df_scaled)
            probability1 = prediction[0][0]
            
            print(probability1, probability2, probability3)
            probability = (probability1 + probability2 + probability3) / 3
        # Hiển thị kết quả
        st.subheader("Kết quả dự đoán 📈")
        st.write(f"🔹 Xác suất mắc bệnh tim: **{probability:.2f}**")

        if probability > 0.4:
            st.error("⚠️ Cảnh báo: Có khả năng bạn mắc bệnh tim. Hãy tham khảo ý kiến bác sĩ!")
        else:
            st.success("🎉 Chúc mừng! Bạn có nguy cơ thấp mắc bệnh tim.")

        st.caption("(*Dự đoán mang tính tham khảo, không thay thế chẩn đoán y tế.)")
