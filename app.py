import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================
# CONFIGURATION & LOADING
# ==========================
st.set_page_config(page_title="INX Future Inc - Employee Performance Predictor", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("best_model_RandomForest.pkl")

model = load_model()

st.title("🧠 INX Future Inc — Employee Performance Prediction")
st.markdown("""
This tool predicts an **employee's expected performance rating** based on key factors such as department, job role, 
experience, and work-life balance.  
It can be used by HR and hiring teams to **evaluate new candidates** and **improve workforce productivity**.
""")

# ==========================
# SIDEBAR INPUT FORM
# ==========================
st.sidebar.header("Enter Employee Details")
st.sidebar.write("Provide candidate details below for prediction:")

age = st.sidebar.slider("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Background", ["Marketing", "Life Sciences", "Human Resources", "Medical", "Technical Degree", "Other"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.sidebar.selectbox("Department", ["Sales", "Human Resources", "Research & Development"])
job_role = st.sidebar.selectbox("Job Role", ["Manager", "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Human Resources"])
travel = st.sidebar.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
education_level = st.sidebar.slider("Education Level", 1, 5, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
num_companies = st.sidebar.slider("Num Companies Worked", 0, 10, 2)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
last_hike = st.sidebar.slider("Last Salary Hike (%)", 0, 25, 10)
relationship = st.sidebar.slider("Relationship Satisfaction", 1, 4, 3)
experience_total = st.sidebar.slider("Total Work Experience (Years)", 0, 40, 10)
training_times = st.sidebar.slider("Training Times Last Year", 0, 10, 2)
work_life_balance = st.sidebar.slider("Work-Life Balance", 1, 4, 3)
experience_company = st.sidebar.slider("Experience Years at Company", 0, 40, 5)
experience_role = st.sidebar.slider("Experience Years in Current Role", 0, 20, 5)
years_promo = st.sidebar.slider("Years Since Last Promotion", 0, 20, 2)
years_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 4)

# ==========================
# PREDICTION SECTION
# ==========================
input_dict = {
    "Age": age,
    "Gender": gender,
    "EducationBackground": education,
    "MaritalStatus": marital_status,
    "EmpDepartment": department,
    "EmpJobRole": job_role,
    "BusinessTravelFrequency": travel,
    "DistanceFromHome": distance,
    "EmpEducationLevel": education_level,
    "EmpEnvironmentSatisfaction": environment_satisfaction,
    "EmpJobInvolvement": job_involvement,
    "EmpJobSatisfaction": job_satisfaction,
    "NumCompaniesWorked": num_companies,
    "OverTime": overtime,
    "EmpLastSalaryHikePercent": last_hike,
    "EmpRelationshipSatisfaction": relationship,
    "TotalWorkExperienceInYears": experience_total,
    "TrainingTimesLastYear": training_times,
    "EmpWorkLifeBalance": work_life_balance,
    "ExperienceYearsAtThisCompany": experience_company,
    "ExperienceYearsInCurrentRole": experience_role,
    "YearsSinceLastPromotion": years_promo,
    "YearsWithCurrManager": years_manager
}

input_df = pd.DataFrame([input_dict])

st.write("### Employee Profile")
st.dataframe(input_df)

if st.button("🔍 Predict Performance"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"🏆 Predicted Performance Rating: **{prediction}** (out of 4)")
    
    # Recommendations based on model insights
    st.markdown("### 💡 Recommendations:")
    
    if prediction <= 2:
        st.warning("""
        - Focus on **skill training** and **mentorship** to enhance job satisfaction.  
        - Reassess workload balance and career growth opportunities.  
        - Conduct 1-on-1 sessions to understand blockers.  
        """)
    elif prediction == 3:
        st.info("""
        - Maintain consistent **work-life balance** and provide targeted **career development programs**.  
        - Encourage involvement in cross-departmental projects to increase engagement.  
        """)
    else:
        st.success("""
        - Excellent performer! Encourage continued growth through **leadership training**.  
        - Consider retention incentives — high performers drive company success.  
        """)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("© 2025 INX Future Inc | IABAC Certified Data Science Project | Built with Streamlit 💡")
