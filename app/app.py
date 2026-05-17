import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="Mental Health Early Warning System", layout="wide")

# --- CUSTOM CSS ---

st.markdown("""
    <style>
    /* 🔥 Increase question (radio label) size */
    div[data-testid="stRadio"] > label {
    font-size: 22px !important;
    font-weight: 600;
    }
    
    /* 1. Global Background with Mesh Gradient */
    .stApp {
        background: radial-gradient(at 0% 0%, #e2eafc 0, transparent 50%), 
                    radial-gradient(at 50% 0%, #f8fbff 0, transparent 50%), 
                    radial-gradient(at 100% 0%, #d7e3fc 0, transparent 50%);
        background-attachment: fixed;
    }

    /* 2. Glassmorphism for Containers */
    [data-testid="stVerticalBlock"] > div:has(div[data-testid="stContainer"]) {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
    }

    /* 3. Modern Sidebar with simple lines */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4);
        border-right: 1px solid rgba(0,0,0,0.05);
    }

    /* 4. Elegant Typography & Buttons */
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700; }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        border: none; color: white; font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "personal"

# SIDEBAR PROGRESS TRACKER 
with st.sidebar:
    st.title(" 📝 Health Tracker")
    steps = ["Personal Info", "Anxiety Assessment", "Depression Assessment", "Stress Assessment", "Final Report"]
    page_map = {"personal": 0, "anxiety": 1, "depression": 2, "stress": 3, "result": 4}
    
    current_idx = page_map.get(st.session_state.page, 0)
    
    st.write("---")
    for i, step in enumerate(steps):
        if i == current_idx:
            st.markdown(f"🔵 **{step}** (Current)")
        elif i < current_idx:
            st.markdown(f"✅ {step}")
        else:
            st.markdown(f"⚪ {step}")
    st.write("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational and prescreening purposes only. It is **NOT** a clinical diagnosis. Please consult a professional for medical advice.")
   
# ------------------------------------------------
# QUESTIONNAIRE SCORE MAPS
# ------------------------------------------------

gad_phq_map = {
    "Not at all":0,
    "Several days":1,
    "More than half the days":2,
    "Nearly every day":3
}

pss_map = {
    "Never":0,
    "Almost Never":1,
    "Sometimes":2,
    "Fairly Often":3,
    "Very Often":4
}

# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
@st.cache_resource
def load_and_train_model():

    df = pd.read_csv("mental.csv")

    df = df.dropna().drop_duplicates()
    print(df.head())
    print(df.info())
    print(df.describe())

    cat_cols = [
        'gender',
        'employment_status',
        'Substance_Use',
        'Mood_Stability',
        'Social_Interaction_Level',
        'mental_health_history',
        'seeks_treatment',
        
    ]

    le_dict = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    target_map = {'Low':0,'Medium':1,'High':2}
    inv_target_map = {0:'Low',1:'Medium',2:'High'}

    df['Risk_Level_Num'] = df['mental_health_risk'].map(target_map)

    features = [c for c in df.columns if c not in ['mental_health_risk','Risk_Level_Num']]

    X = df[features]
    y = df['Risk_Level_Num']

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    smote = SMOTE(random_state=42)

    X_train_res,y_train_res = smote.fit_resample(X_train,y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        random_state=42
    )

    model.fit(X_train_res,y_train_res)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print("\n====== MODEL PERFORMANCE ======")
    print("Accuracy:",round(accuracy_score(y_test,y_pred)*100,2),"%")
    print("\nConfusion Matrix:",cm)
    print("\nClassification Report\n")
    print(classification_report(y_test,y_pred))
    print("===============================\n")

    return model,le_dict,features,inv_target_map, cm
rf_model,le_dict,features,inv_target_map, conf_matrix = load_and_train_model()

# ------------------------------------------------
# EARLY WARNING SYSTEM
# ------------------------------------------------
def early_warning_system(model, input_df):
    proba = model.predict_proba(input_df)[0]
    high_risk_percent = proba[2] * 100 

    if high_risk_percent >= 75:
        return "🔴 CRITICAL WARNING – Extremely high probability of risk. Immediate support needed.", high_risk_percent
    elif high_risk_percent >= 45:
        return "🟠 MODERATE WARNING – Signs of risk detected. Early intervention advised.", high_risk_percent
    elif high_risk_percent >= 20:
        return "🟡 MONITOR – Low probability signs detected. Regular check-ins recommended.", high_risk_percent
    else:
        return "🟢 STABLE – No significant risk patterns detected.", high_risk_percent

# ------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------

def generate_recommendations(data):

    recs=[]

    if data["sleep_hours"] < 6:
        recs.append("Improve sleep habits (7–8 hours recommended).")

    if data["Screen_Time_Hours"] > 5:
        recs.append("Reduce excessive screen time.")

    if data["physical_activity_days"] < 3:
        recs.append("Increase physical activity during the week.")

    if data["stress_level"] > 5:
        recs.append("Practice meditation or stress management techniques.")

    if data["depression_score"] > 9:
        recs.append("Consider consulting a mental health professional.")

    if data["anxiety_score"] > 9:
        recs.append("Try relaxation or breathing exercises.")

    if data["productivity_score"] < 50:
        recs.append("Improve work-life balance and daily planning.")

    if len(recs)==0:
        recs.append("Maintain your healthy routine.")

    return recs

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page="personal"

# ------------------------------------------------
# PAGE 1 PERSONAL + LIFESTYLE
# ------------------------------------------------
if st.session_state.page=="personal":
    st.title("🧠 Mental Health Prescreening Tool")
    st.divider()

    with st.container(border=True):

        st.subheader("Personal Data")
        col1,col2 = st.columns(2)
        with col1:
            age = st.number_input("Age",15,100,25)

            gender = st.selectbox("Gender",["Male","Female","Prefer not to say"])

        with col2:
            
            employment = st.selectbox("Employment Status",["Student","Employed","Unemployed","Self-employed"])

    with st.container(border=True):

        st.subheader("Lifestyle & Behavioural Data")

        col3,col4=st.columns(2)

        with col3:
            sleep_hours = st.number_input("Sleep Hours",1,12,7)
            physical_activity = st.number_input("Physical Activity Days per Week",0,7,3)
            
        with col4:
            Screen_Time_Hours = st.number_input("Screen Time Hours",0,24,4)
            Substance_use = st.selectbox("Substance Use",["No","Yes"])
            
    with st.container(border=True):

        st.subheader("Dynamic Features")

        col5,col6=st.columns(2)

        with col5:
            Mood_Stability = st.selectbox("Mood Stability",["Stable","Fluctuating"])
            Social_Interaction_Level = st.selectbox("Social Interaction Level",["Low","Medium","High"])
            weekly_stress_change = st.slider("Weekly Stress Change",-3,3,0)
            
        with col6:
            mental_history = st.selectbox("Mental Health History",["No","Yes"])
            treatment = st.selectbox("Seeks Treatment",["No","Yes"])
            productivity_score = st.number_input("Productivity Score",0,100,70)
            
    if st.button("Next → Anxiety Assessment"):

        st.session_state.personal={
            "age":age,
            "gender":gender,
            "employment_status":employment,
            "sleep_hours":sleep_hours,
            "physical_activity_days":physical_activity,
            "Screen_Time_Hours":Screen_Time_Hours,
            "Substance_Use":Substance_use,
            "Mood_Stability":Mood_Stability,
            "Social_Interaction_Level":Social_Interaction_Level,
            "Weekly_Stress_Change":weekly_stress_change,
            "productivity_score":productivity_score,
            "mental_health_history":mental_history,
            "seeks_treatment":treatment
        }

        st.session_state.page="anxiety"
        st.rerun()
# ------------------------------------------------
# PAGE 2 GAD7
# ------------------------------------------------

elif st.session_state.page=="anxiety":

    st.title("Anxiety Assessment (GAD-7)")

    st.write("Over the last 2 weeks, how often have you been bothered by the following problems?")

    opt=list(gad_phq_map.keys())

    with st.container(border=True):
        q1=st.radio("1. Feeling nervous, anxious, or on edge?",opt)

    with st.container(border=True):
        q2=st.radio("2. Not being able to stop or control worrying?",opt)

    with st.container(border=True):
        q3=st.radio("3. Worrying too much about different things?",opt)

    with st.container(border=True):
        q4=st.radio("4. Trouble relaxing?",opt)

    with st.container(border=True):
        q5=st.radio("5. Being so restless that it is hard to sit still?",opt)

    with st.container(border=True):
        q6=st.radio("6. Becoming easily annoyed or irritated?",opt)

    with st.container(border=True):
        q7=st.radio("7. Feeling afraid as if something awful might happen?",opt)

    if st.button("Next → Depression Assessment"):

        anxiety_score=sum([
            gad_phq_map[q1],gad_phq_map[q2],gad_phq_map[q3],
            gad_phq_map[q4],gad_phq_map[q5],gad_phq_map[q6],
            gad_phq_map[q7]
        ])

        st.session_state.anxiety_score=anxiety_score
        st.session_state.page="depression"
        st.rerun()

# ------------------------------------------------
# PAGE 3 PHQ9
# ------------------------------------------------

elif st.session_state.page=="depression":

    st.title("Depression Assessment (PHQ-9)")

    st.write("Over the last 2 weeks, how often have you been bothered by the following problems?")

    opt=list(gad_phq_map.keys())

    with st.container(border=True):
        q1=st.radio("1. Little interest or pleasure in doing things",opt)

    with st.container(border=True):
        q2=st.radio("2. Feeling down, depressed, or hopeless",opt)

    with st.container(border=True):
        q3=st.radio("3. Trouble falling or staying asleep, or sleeping too much",opt)

    with st.container(border=True):
        q4=st.radio("4. Feeling tired or having little energy",opt)

    with st.container(border=True):
        q5=st.radio("5. Poor appetite or overeating",opt)

    with st.container(border=True):
        q6=st.radio("6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",opt)

    with st.container(border=True):
        q7=st.radio("7. Trouble concentrating on things, such as reading the newspaper or watching television",opt)

    with st.container(border=True):
        q8=st.radio("8. Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",opt)

    with st.container(border=True):
        q9=st.radio("9. Thoughts that you would be better off dead, or of hurting yourself in some way",opt)

    if st.button("Next → Stress Assessment"):

        depression_score=sum([
            gad_phq_map[q1],gad_phq_map[q2],gad_phq_map[q3],
            gad_phq_map[q4],gad_phq_map[q5],gad_phq_map[q6],
            gad_phq_map[q7],gad_phq_map[q8],gad_phq_map[q9]
        ])

        st.session_state.depression_score=depression_score
        st.session_state.page="stress"
        st.rerun()

# ------------------------------------------------
# PAGE 4 PSS10
# ------------------------------------------------

elif st.session_state.page=="stress":

    st.title("Stress Assessment (PSS-10)")
    st.write("In the last month, how often have you felt or thought the following?")

    opt=list(pss_map.keys())

    with st.container(border=True):
        q1=st.radio("1. In the last month, how often have you been upset because of something that happened unexpectedly?",opt)

    with st.container(border=True):
        q2=st.radio("2. In the last month, how often have you felt that you were unable to control the important things in your life?",opt)

    with st.container(border=True):
        q3=st.radio("3. In the last month, how often have you felt nervous and stressed?",opt)

    with st.container(border=True):
        q4=st.radio("4. In the last month, how often have you felt confident about your ability to handle your personal problems?",opt)

    with st.container(border=True):
        q5=st.radio("5.In the last month, how often have you felt that things were going your way?",opt)

    with st.container(border=True):
        q6=st.radio("6. In the last month, how often have you found that you could not cope with all the things that you had to do?",opt)

    with st.container(border=True):
        q7=st.radio("7. In the last month, how often have you been able to control irritations in your life?",opt)

    with st.container(border=True):
        q8=st.radio("8. In the last month, how often have you felt that you were on top of things?",opt)

    with st.container(border=True):
        q9=st.radio("9. In the last month, how often have you been angered because of things that happened that were outside of your control?",opt)

    with st.container(border=True):
        q10=st.radio("10. In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",opt)

    if st.button("Generate Prediction"):

        s1=pss_map[q1]
        s2=pss_map[q2]
        s3=pss_map[q3]

        s4=4-pss_map[q4]
        s5=4-pss_map[q5]

        s6=pss_map[q6]

        s7=4-pss_map[q7]
        s8=4-pss_map[q8]

        s9=pss_map[q9]
        s10=pss_map[q10]

        stress_score_raw=sum([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10])
         # Convert to dataset scale (0-10)
        stress_level = round(stress_score_raw / 4,2)

        st.session_state.stress_level=stress_level
        st.session_state.page="result"
        st.rerun()
# ------------------------------------------------
# RESULT PAGE
# ------------------------------------------------

elif st.session_state.page=="result":

    st.title("📊 Mental Health Prediction Report")

    data=st.session_state.personal

    input_data={
        **data,
        "anxiety_score":st.session_state.anxiety_score,
        "depression_score":st.session_state.depression_score,
        "stress_level":st.session_state.stress_level,
    }

    df=pd.DataFrame([input_data])

    for col,le in le_dict.items():
        df[col]=le.transform(df[col])

    pred=rf_model.predict(df[features])[0]
    # Risk probability
    proba = rf_model.predict_proba(df[features])[0]

    prob_low = round(proba[0]*100,2)
    prob_med = round(proba[1]*100,2)
    prob_high = round(proba[2]*100,2)

    label=inv_target_map[pred]

    feat_importances=pd.Series(
        rf_model.feature_importances_,
        index=features
    ).sort_values(ascending=False).head(5)

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Predicted Risk Level")
        st.subheader("Risk Probability")

        st.write(f"Low Risk Probability: {prob_low}%")
        st.write(f"Medium Risk Probability: {prob_med}%")
        st.write(f"High Risk Probability: {prob_high}%")

        if label=="High":
            st.error("High Risk")
        elif label=="Medium":
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")
        warning_text, risk_score = early_warning_system(rf_model, df[features])

        st.subheader("Early Warning Status")
        st.info(warning_text)
        
    with col2:

        st.subheader("Recommendations")
        recs=generate_recommendations(input_data)
        for r in recs:
            st.write("•",r)

    st.divider()
    st.subheader("Explainable AI (Model Explanation)")
    explainer = shap.TreeExplainer(rf_model)
    user_input_array = df[features].iloc[0,:].values
    shap_values = explainer.shap_values(df[features])
    try:
        class_idx = int(pred)
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0,:]
            bv = explainer.expected_value[class_idx]
        else:
            sv = shap_values[0,:,class_idx]
            bv = explainer.expected_value[class_idx]
        fig_shap = shap.force_plot(
            bv,
            sv,
            user_input_array,
            feature_names=features,
            matplotlib=True,
            show=False
        )
            
        st.pyplot(fig_shap, bbox_inches='tight')
        plt.clf() 
    except Exception as e:
        st.warning("Alternative view: Key contributing factors")
        st.bar_chart(feat_importances)
    st.divider()

    st.subheader("Key Contributing Factors")
    for i,(feat,val) in enumerate(feat_importances.items()):
        st.write(f"{i+1}. {feat}")

    if st.button("Restart Assessment"):
        st.session_state.page="personal"
        st.rerun()

