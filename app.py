import joblib
import pandas as pd
import streamlit as st

# LOAD ARTIFACTS
num_cols = joblib.load("num_cols.pkl")
scaler = joblib.load("scaler.pkl")
onehot_columns = joblib.load("onehot_columns.pkl")
final_ridge = joblib.load("final_ridge_model.pkl")

# PREDICTION FUNCTION
def predict_score(age, study_hours, class_attendance, sleep_hours,
                  gender, course, internet_access, sleep_quality,
                  study_method, facility_rating, exam_difficulty):

    data = {
        "age": age,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "sleep_hours": sleep_hours,
        "gender": gender,
        "course": course,
        "internet_access": internet_access,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }

    df_input = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    df_encoded = df_encoded.reindex(columns=onehot_columns, fill_value=0)
    df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

    pred = final_ridge.predict(df_encoded)[0]
    pred = max(0, min(pred, 100))
    return round(float(pred), 2)

# PAGE CONFIG
st.set_page_config(
    page_title="Exam Score Predictor",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

* {
    font-family: 'Montserrat', sans-serif !important;
}

/* App background */
.stApp {
    background-color: #F1EFEC;
}

/* Headings */
h1 {
    color: #123458 !important;
    font-weight: 700 !important;
}

/* Labels */
label {
    color: #123458 !important;
    font-weight: 600 !important;
}

/* FORM CARD */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 36px;
    border-radius: 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.08);
}

/* SLIDERS */
.stSlider > div {
    padding-top: 8px;
}

/* SELECTBOX (DARK INPUT) */
div[data-baseweb="select"] {
    background-color: #123458 !important;
    border-radius: 14px !important;
    border: none !important;
}

div[data-baseweb="select"] span {
    color: #F1EFEC !important;
    font-weight: 500 !important;
}

/* Dropdown menu */
ul[data-baseweb="menu"] {
    background-color: #123458 !important;
    border-radius: 14px !important;
    padding: 6px !important;
}

/* Dropdown item text */
ul[data-baseweb="menu"] li {
    color: #F1EFEC !important;
    font-weight: 500 !important;
}

/* Hover effect (biar kebaca) */
ul[data-baseweb="menu"] li:hover {
    background-color: #1f4f82 !important;
    color: #FFFFFF !important;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(135deg, #123458, #1a4a7a) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 14px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    width: 100%;
    border: none !important;
    box-shadow: 0 8px 24px rgba(18,52,88,0.35);
}

.stButton button:hover {
    transform: translateY(-2px);
}

/* RESULT BOX */
.result-box {
    margin-top: 32px;
    background: linear-gradient(135deg, #123458, #0e2c4f);
    padding: 36px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 16px 40px rgba(0,0,0,0.3);
}

.result-title {
    color: #D4C9BE;
    font-size: 18px;
    margin-bottom: 8px;
}

.result-score {
    font-size: 56px;
    font-weight: 700;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("Exam Score Predictor")
st.markdown(
    "Estimate student exam scores based on study habits and lifestyle factors."
)

st.markdown("<br>", unsafe_allow_html=True)

# INPUT FORM
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 24, 17)
        study_hours = st.slider("Study Hours per Day", 0, 8, 0)
        class_attendance = st.slider("Class Attendance (%)", 40, 100, 40)
        sleep_hours = st.slider("Sleep Hours per Day", 4, 10, 4)

        gender = st.selectbox("Gender", ["male", "female", "other"], index=None, placeholder="Choose")

        course = st.selectbox(
            "Course",
            ["diploma", "bca", "b.sc", "b.tech", "bba", "ba", "b.com"],
            index=None,
            placeholder="Choose"
        )

    with col2:
        internet_access = st.selectbox("Internet Access", ["yes", "no"], index=None, placeholder="Choose")

        sleep_quality = st.selectbox(
            "Sleep Quality", ["poor", "average", "good"], index=None, placeholder="Choose"
        )

        study_method = st.selectbox(
            "Study Method",
            ["coaching", "online videos", "self-study"],
            index=None,
            placeholder="Choose"
        )

        facility_rating = st.selectbox(
            "Facility Rating", ["low", "medium", "high"], index=None, placeholder="Choose"
        )

        exam_difficulty = st.selectbox(
            "Exam Difficulty", ["easy", "moderate", "hard"], index=None, placeholder="Choose"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Predict Score")

# OUTPUT
if submit:
    if None in [
        gender, course, internet_access,
        sleep_quality, study_method,
        facility_rating, exam_difficulty
    ]:
        st.warning("Please complete all fields before predicting.")
    else:
        score = predict_score(
            age, study_hours, class_attendance, sleep_hours,
            gender, course, internet_access, sleep_quality,
            study_method, facility_rating, exam_difficulty
        )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">Predicted Exam Score</div>
                <div class="result-score">{score}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
