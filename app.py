import streamlit as st
import pandas as pd
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Diabetes Predictor Suite", page_icon="🩺", layout="wide")

# Set consistent plotting style for crisp images
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 50, 'font.size': 10})

# --- 1. INITIALIZE HISTORY DATA IN SESSION STATE ---
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# --- 2. SAFELY LOAD MODEL ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.sidebar.error("🚨 Missing 'logistic_model.pkl' or 'scaler.pkl'!")
        return None, None

model, scaler = load_assets()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Companionship Menu")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to System Engine:", ["Diabetes Predictor", "History Log", "Live Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Populate records in the Predictor tab to unlock live charts in the Dashboard.")

# Helper function to convert matplotlib figures to bytes for downloads
def convert_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ==========================================
# PAGE 1: DIABETES PREDICTOR
# ==========================================
if page == "Diabetes Predictor":
    st.title("🩺 Medical Assistant: Diabetes Predictor")
    st.write("Adjust patient clinical parameters below to evaluate diagnostic results.")
    st.markdown("---")

    if model is not None and scaler is not None:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Patient Gender", ["Female", "Male", "Other"])
            age = st.slider("Patient Age", 1.0, 100.0, 45.0, 1.0)
            hypertension = st.selectbox("Hypertension Status", ["No Hypertension", "Has Hypertension"])
            heart_disease = st.selectbox("Heart Disease Status", ["No Heart Disease", "Has Heart Disease"])
        with col2:
            smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
            bmi = st.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0, 0.1)
            hba1c = st.slider("HbA1c Level (%)", 3.0, 10.0, 5.5, 0.1)
            glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 120, 5)

        if st.button("Predict Diabetes Status", use_container_width=True):
            # Convert binary flags to 0/1 matching training standards
            hyp_binary = 1 if hypertension == "Has Hypertension" else 0
            hd_binary = 1 if heart_disease == "Has Heart Disease" else 0

            # Create initial raw DataFrame for user input
            raw_input = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'hypertension': hyp_binary,
                'heart_disease': hd_binary,
                'smoking_history': smoking_history,
                'bmi': bmi,
                'HbA1c_level': hba1c,
                'blood_glucose_level': glucose
            }])

            # Perform One-Hot Encoding
            encoded_input = pd.get_dummies(raw_input, columns=['gender', 'smoking_history'], drop_first=True)

            # Pipeline alignment fix
            expected_training_columns = [
                'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                'gender_Male', 'gender_Other', 
                'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                'smoking_history_never', 'smoking_history_not current'
            ]
            aligned_input = encoded_input.reindex(columns=expected_training_columns, fill_value=0)

            # Transform & Predict
            scaled_input = scaler.transform(aligned_input)
            prediction = int(model.predict(scaled_input)[0])


            # Map the visual outputs
            status_text = "POSITIVE (1)" if prediction == 1 else "NEGATIVE (0)"
            
            st.markdown("---")
            st.subheader("📊 Diagnostic Summary Result")
            if prediction == 1:
                st.error(f"🚨 **Diabetes Status**: {status_text}")
            else:
                st.success(f"✅ **Diabetes Status**: {status_text}")

            # --- SAVE RECORD TO HISTORY ---
            new_record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Gender": gender,
                "Age": age,
                "Hypertension": hyp_binary,
                "Heart Disease": hd_binary,
                "Smoking History": smoking_history,
                "BMI": bmi,
                "HbA1c Level (%)": hba1c,
                "Blood Glucose": glucose,
                "Prediction Result": status_text
            }
            st.session_state["prediction_history"].append(new_record)
            st.toast("Patient data successfully logged to history matrix!", icon="📝")

# ==========================================
# PAGE 2: HISTORY LOG
# ==========================================
elif page == "History Log":
    st.title("📜 Patient Diagnosis History Log")
    st.write("View, manage, and download all evaluation logs captured during this browser session.")
    st.markdown("---")

    if len(st.session_state["prediction_history"]) == 0:
        st.info("No records found. Run a prediction on the dashboard page to populate this log.")
    else:
        history_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_data,
                file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with btn_col2:
            if st.button("🗑️ Clear Evaluation History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

# ==========================================
# PAGE 3: LIVE DASHBOARD
# ==========================================
elif page == "Live Dashboard":
    st.title("📊 Clinical Analytics Dashboard")
    st.write("Real-time clinical insights generated instantly from session patient entries.")
    st.markdown("---")

    # Automated check for history clearing / empty states
    if len(st.session_state["prediction_history"]) == 0:
        st.warning("⚠️ Dashboard Reset: No active medical records detected. Enter data in the 'Diabetes Predictor' to build live visuals.")
    else:
        # Fetch active records from session cache
        dash_df = pd.DataFrame(st.session_state["prediction_history"])
        
        # Calculate Core Metrics
        total_examined = len(dash_df)
        total_positive = len(dash_df[dash_df["Prediction Result"] == "POSITIVE (1)"])
        total_negative = len(dash_df[dash_df["Prediction Result"] == "NEGATIVE (0)"])

        # Display Top Summary KPI Scorecards
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="👥 Total Patients Examined", value=total_examined)
        m_col2.metric(label="🚨 Total Positive (Diabetes)", value=total_positive, delta=f"{(total_positive/total_examined)*100:.1f}% Share", delta_color="inverse")
        m_col3.metric(label="✅ Total Negative (Healthy)", value=total_negative, delta=f"{(total_negative/total_examined)*100:.1f}% Share")
        
        st.markdown("---")
        st.subheader("📈 Specialized Diagnostic Visualizations")
        st.write("Click the download button under any chart to save a high-resolution PNG image directly to your local computer.")
        st.markdown("<br>", unsafe_allow_html=True)

                # -----------------------------------------------------------------
        # ROW 1: Demographic Breakdown
        # -----------------------------------------------------------------
        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            st.markdown("#### 1. Gender Distribution")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            gender_counts = dash_df["Gender"].value_counts()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=90)
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "1_gender_distribution.png", "image/png", use_container_width=True)

        with row1_col2:
            st.markdown("#### 2. Diabetes Status by Smoking")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sns.countplot(data=dash_df, x='Smoking History', hue='Prediction Result', palette='Set2', ax=ax)
            ax.set_title('Diabetes Status by Smoking History')
            ax.tick_params(axis='x', rotation=35)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "2_diabetes_by_smoking.png", "image/png", use_container_width=True)

        with row1_col3:
            st.markdown("#### 3. Diabetes by Heart Disease")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            # FIX: Removed the static set_xticks to let the categorical data layout scale dynamically
            sns.countplot(data=dash_df, x='Heart Disease', hue='Prediction Result', palette='coolwarm', ax=ax)
            ax.set_title('Diabetes Distribution by Heart Disease Status')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "3_diabetes_by_heart_disease.png", "image/png", use_container_width=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------
        # ROW 2: Continuous Vital Trackers
        # -----------------------------------------------------------------
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            st.markdown("#### 4. Age Distribution Patterns")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            if dash_df['Age'].nunique() > 1:
                sns.kdeplot(data=dash_df, x='Age', hue='Prediction Result', fill=True, palette='Set1', common_norm=False, ax=ax)
            else:
                sns.histplot(data=dash_df, x='Age', hue='Prediction Result', multiple="stack", palette='Set1', ax=ax)
            ax.set_title('Age Distribution Patterns by Diabetes Status')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "4_age_distribution.png", "image/png",             use_container_width=True)

        with row2_col2:
            st.markdown("#### 5. Diabetes Counts by Gender")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sns.countplot(data=dash_df, x='Gender', hue='Prediction Result', palette='Set1', ax=ax)
            ax.set_title('Diabetes Patient Counts by Gender')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "5_diabetes_counts_by_gender.png", "image/png", use_container_width=True)

        with row2_col3:
            st.markdown("#### 6. Diabetes by Hypertension")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            # FIX: Removed static set_xticks constraint so it scales correctly with data entries
            sns.countplot(data=dash_df, x='Hypertension', hue='Prediction Result', palette='muted', ax=ax)
            ax.set_title('Diabetes Counts grouped by Hypertension')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "6_diabetes_by_hypertension.png", "image/png", use_container_width=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------
        # ROW 3: Clinical Lab Biomarkers
        # -----------------------------------------------------------------
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        with row3_col1:
            st.markdown("#### 7. BMI Distribution Patterns")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            if dash_df['BMI'].nunique() > 1:
                sns.kdeplot(data=dash_df, x='BMI', hue='Prediction Result', fill=True, palette='Set2', common_norm=False, ax=ax)
            else:
                sns.histplot(data=dash_df, x='BMI', hue='Prediction Result', multiple="stack", palette='Set2', ax=ax)
            ax.set_title('BMI Distribution Patterns by Diabetes Status')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "7_bmi_distribution.png", "image/png", use_container_width=True)

        with row3_col2:
            st.markdown("#### 8. Blood Glucose Distribution")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            if dash_df['Blood Glucose'].nunique() > 1:
                sns.kdeplot(data=dash_df, x='Blood Glucose', hue='Prediction Result', fill=True, palette='rocket', common_norm=False, ax=ax)
            else:
                sns.histplot(data=dash_df, x='Blood Glucose', hue='Prediction Result', multiple="stack", palette='rocket', ax=ax)
            ax.set_title('Blood Glucose Distribution by Diabetes Status')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "8_blood_glucose_distribution.png", "image/png", use_container_width=True)

        with row3_col3:
            st.markdown("#### 9. Average HbA1c Level")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sns.barplot(data=dash_df, x='Prediction Result', y='HbA1c Level (%)', hue='Prediction Result', legend=False, errorbar=None, palette='flare', ax=ax)
            ax.set_title('Average HbA1c Level by Diabetes Status')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("📥 Save Image", convert_fig_to_bytes(fig), "9_average_hba1c.png", "image/png", use_container_width=True)