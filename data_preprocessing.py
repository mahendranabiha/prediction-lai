import joblib
import numpy as np

# Load joblib file dari scaler fitur numerik
scaler_Previous_qualification_grade = joblib.load(
    "preprocessing/scaler_Previous qualification (grade).joblib"
)
scaler_Admission_grade = joblib.load("preprocessing/scaler_Admission grade.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load(
    "preprocessing/scaler_Curricular units 1st sem (approved).joblib"
)
scaler_Curricular_units_1st_sem_grade = joblib.load(
    "preprocessing/scaler_Curricular units 1st sem (grade).joblib"
)
scaler_Curricular_units_2nd_sem_approved = joblib.load(
    "preprocessing/scaler_Curricular units 2nd sem (approved).joblib"
)
scaler_Curricular_units_2nd_sem_grade = joblib.load(
    "preprocessing/scaler_Curricular units 2nd sem (grade).joblib"
)

# Load joblib file dari encoder fitur kategorik
encoder_Application_mode = joblib.load("preprocessing/encoder_Application mode.joblib")
encoder_Course = joblib.load("preprocessing/encoder_Course.joblib")
encoder_Previous_qualification = joblib.load(
    "preprocessing/encoder_Previous qualification.joblib"
)
encoder_Nacionality = joblib.load("preprocessing/encoder_Nacionality.joblib")
encoder_Mother_qualification = joblib.load(
    "preprocessing/encoder_Mother's qualification.joblib"
)
encoder_Father_qualification = joblib.load(
    "preprocessing/encoder_Father's qualification.joblib"
)
encoder_Mother_occupation = joblib.load(
    "preprocessing/encoder_Mother's occupation.joblib"
)
encoder_Father_occupation = joblib.load(
    "preprocessing/encoder_Father's occupation.joblib"
)
encoder_Displaced = joblib.load("preprocessing/encoder_Displaced.joblib")
encoder_Tuition_fees_up_to_date = joblib.load(
    "preprocessing/encoder_Tuition fees up to date.joblib"
)
encoder_Scholarship_holder = joblib.load(
    "preprocessing/encoder_Scholarship holder.joblib"
)


def transform_data(DataFrame):
    # Menyalin DataFrame untuk normasilasi dan encoding label: transform_df
    transform_df = DataFrame.copy()

    # Normalisasi fitur numerik
    transform_df["Previous qualification (grade)"] = (
        scaler_Previous_qualification_grade.transform(
            np.asanyarray(transform_df["Previous qualification (grade)"]).reshape(-1, 1)
        )
    )
    transform_df["Admission grade"] = scaler_Admission_grade.transform(
        np.asanyarray(transform_df["Admission grade"]).reshape(-1, 1)
    )
    transform_df["Curricular units 1st sem (approved)"] = (
        scaler_Curricular_units_1st_sem_approved.transform(
            np.asanyarray(transform_df["Curricular units 1st sem (approved)"]).reshape(
                -1, 1
            )
        )
    )
    transform_df["Curricular units 1st sem (grade)"] = (
        scaler_Curricular_units_1st_sem_grade.transform(
            np.asanyarray(transform_df["Curricular units 1st sem (grade)"]).reshape(
                -1, 1
            )
        )
    )
    transform_df["Curricular units 2nd sem (approved)"] = (
        scaler_Curricular_units_2nd_sem_approved.transform(
            np.asanyarray(transform_df["Curricular units 2nd sem (approved)"]).reshape(
                -1, 1
            )
        )
    )
    transform_df["Curricular units 2nd sem (grade)"] = (
        scaler_Curricular_units_2nd_sem_grade.transform(
            np.asanyarray(transform_df["Curricular units 2nd sem (grade)"]).reshape(
                -1, 1
            )
        )
    )

    # Encoding label fitur kategorik
    transform_df["Application mode"] = encoder_Application_mode.transform(
        transform_df["Application mode"]
    )
    transform_df["Course"] = encoder_Course.transform(transform_df["Course"])
    transform_df["Previous qualification"] = encoder_Previous_qualification.transform(
        transform_df["Previous qualification"]
    )
    transform_df["Nacionality"] = encoder_Nacionality.transform(
        transform_df["Nacionality"]
    )
    transform_df["Mother's qualification"] = encoder_Mother_qualification.transform(
        transform_df["Mother's qualification"]
    )
    transform_df["Father's qualification"] = encoder_Father_qualification.transform(
        transform_df["Father's qualification"]
    )
    transform_df["Mother's occupation"] = encoder_Mother_occupation.transform(
        transform_df["Mother's occupation"]
    )
    transform_df["Father's occupation"] = encoder_Father_occupation.transform(
        transform_df["Father's occupation"]
    )
    transform_df["Displaced"] = encoder_Displaced.transform(transform_df["Displaced"])
    transform_df["Tuition fees up to date"] = encoder_Tuition_fees_up_to_date.transform(
        transform_df["Tuition fees up to date"]
    )
    transform_df["Scholarship holder"] = encoder_Scholarship_holder.transform(
        transform_df["Scholarship holder"]
    )

    # Mengembalikan hasil transformasi data
    return transform_df
