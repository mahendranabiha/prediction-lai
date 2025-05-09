import streamlit as st
import pandas as pd

from data_preprocessing import (
    transform_data,
    encoder_Application_mode,
    encoder_Course,
    encoder_Previous_qualification,
    encoder_Nacionality,
    encoder_Mother_qualification,
    encoder_Father_qualification,
    encoder_Mother_occupation,
    encoder_Father_occupation,
    encoder_Displaced,
    encoder_Tuition_fees_up_to_date,
    encoder_Scholarship_holder,
)
from prediction import predict_data

st.header("Student Dropout Prediction")

df_example = pd.DataFrame(
    {
        "Nacionality": ["Brazilian"],
        "Father's qualification": [
            "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv."
        ],
        "Father's occupation": [
            "Installation and Machine Operators and Assembly Workers"
        ],
        "Mother's qualification": [
            "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv."
        ],
        "Mother's occupation": [
            "Personal Services, Security and Safety Workers and Sellers"
        ],
        "Displaced": ["yes"],
        "Tuition fees up to date": ["yes"],
        "Scholarship holder": ["no"],
        "Previous qualification": ["Secondary education"],
        "Previous qualification (grade)": [137.0],
        "Admission grade": [137.4],
        "Application mode": ["3rd phase - general contingent"],
        "Course": ["Social Service"],
        "Curricular units 1st sem (approved)": [1],
        "Curricular units 1st sem (grade)": [12.0],
        "Curricular units 2nd sem (approved)": [2],
        "Curricular units 2nd sem (grade)": [11.0],
    }
)

with st.expander("View the student data example (enrolled)"):
    st.dataframe(data=df_example, width=800, height=10)

df = pd.DataFrame()

st.subheader("Student Demographics Data")
nacionality = st.selectbox(
    label="Nacionality", options=encoder_Nacionality.classes_, index=1
)
df["Nacionality"] = [nacionality]
col1, col2 = st.columns(2)
with col1:
    father_qualification = st.selectbox(
        label="Father's qualification",
        options=encoder_Father_qualification.classes_,
        index=1,
    )
    df["Father's qualification"] = [father_qualification]
with col2:
    father_occupation = st.selectbox(
        label="Father's occupation", options=encoder_Father_occupation.classes_, index=1
    )
    df["Father's occupation"] = [father_occupation]
col1, col2 = st.columns(2)
with col1:
    mother_qualification = st.selectbox(
        label="Mother's qualification",
        options=encoder_Mother_qualification.classes_,
        index=1,
    )
    df["Mother's qualification"] = [mother_qualification]
with col2:
    mother_occupation = st.selectbox(
        label="Mother's occupation", options=encoder_Mother_occupation.classes_, index=1
    )
    df["Mother's occupation"] = [mother_occupation]

st.subheader("Student Financial Data")
col1, col2, col3 = st.columns(3)
with col1:
    displaced = st.selectbox(
        label="Displaced", options=encoder_Displaced.classes_, index=1
    )
    df["Displaced"] = [displaced]
with col2:
    tuition_fees_up_to_date = st.selectbox(
        label="Tuition fees up to date",
        options=encoder_Tuition_fees_up_to_date.classes_,
        index=1,
    )
    df["Tuition fees up to date"] = [tuition_fees_up_to_date]
with col3:
    scholarship_holder = st.selectbox(
        label="Scholarship holder", options=encoder_Scholarship_holder.classes_, index=1
    )
    df["Scholarship holder"] = [scholarship_holder]

st.subheader("Student Academic History Data")
col1, col2, col3 = st.columns(3)
with col1:
    previous_qualification = st.selectbox(
        label="Previous qualification",
        options=encoder_Previous_qualification.classes_,
        index=1,
    )
    df["Previous qualification"] = [previous_qualification]
with col2:
    previous_qualification_grade = float(
        st.number_input(label="Previous qualification (grade)", value=100.0)
    )
    df["Previous qualification (grade)"] = [previous_qualification_grade]
with col3:
    admission_grade = float(st.number_input(label="Admission grade", value=141.5))
    df["Admission grade"] = [admission_grade]
col1, col2 = st.columns(2)
with col1:
    application_mode = st.selectbox(
        label="Application mode", options=encoder_Application_mode.classes_, index=1
    )
    df["Application mode"] = [application_mode]
with col2:
    course = st.selectbox(label="Course", options=encoder_Course.classes_, index=1)
    df["Course"] = [course]
col1, col2 = st.columns(2)
with col1:
    curricular_units_1st_sem_approved = int(
        st.number_input(label="Curricular units 1st sem (approved)", value=5)
    )
    df["Curricular units 1st sem (approved)"] = [curricular_units_1st_sem_approved]
with col2:
    curricular_units_1st_sem_grade = float(
        st.number_input(label="Curricular units 1st sem (grade)", value=12.33)
    )
    df["Curricular units 1st sem (grade)"] = [curricular_units_1st_sem_grade]
col1, col2 = st.columns(2)
with col1:
    curricular_units_2nd_sem_approved = int(
        st.number_input(label="Curricular units 2nd sem (approved)", value=6)
    )
    df["Curricular units 2nd sem (approved)"] = [curricular_units_2nd_sem_approved]
with col2:
    curricular_units_2nd_sem_grade = float(
        st.number_input(label="Curricular units 2nd sem (grade)", value=13.0)
    )
    df["Curricular units 2nd sem (grade)"] = [curricular_units_2nd_sem_grade]

with st.expander("View the raw data"):
    st.dataframe(data=df, width=800, height=10)

if st.button("Predict"):
    new_df = transform_data(DataFrame=df)
    with st.expander("View the transformed data"):
        st.dataframe(data=new_df, width=800, height=10)
    predict = predict_data(new_df)
    st.write("Siswa Berpotensi: {}".format(predict[0]))
