import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/Telco_churn.csv")


model = load_model()
df = load_data()

df.columns = df.columns.str.strip()


# ============================================================
# DATA PREPARATION
# ============================================================

drop_columns = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)


# ============================================================
# ENCODERS
# ============================================================

categorical_columns = []

for col in df.columns:
    if df[col].dtype == "object":
        categorical_columns.append(col)


encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    encoder.fit(
        df[col].astype(str)
    )

    encoders[col] = encoder


feature_columns = [
    col for col in df.columns
    if col != "Churn Value"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📡 Churn Predictor")

    st.caption(
        "Machine Learning Application"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔮 Predict Churn",
            "📊 Analytics",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption(
        "Random Forest Classifier"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title("📡 Customer Churn Prediction")

    st.write(
        "An interactive machine learning application "
        "that predicts whether a telecom customer is "
        "likely to churn."
    )

    st.divider()

    total_customers = len(df)

    churned_customers = int(
        df["Churn Value"].sum()
    )

    active_customers = (
        total_customers - churned_customers
    )

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )


    with col3:
        st.metric(
            "Active Customers",
            f"{active_customers:,}"
        )


    with col4:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%"
        )


    st.divider()

    st.subheader("🎯 Project Overview")

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### Objective")

        st.write(
            "Identify telecom customers who are likely "
            "to leave the service using machine learning."
        )


    with col2:

        st.markdown("### Model")

        st.write(
            "Random Forest Classifier trained on "
            "customer demographic, service and billing data."
        )


    st.divider()

    st.subheader("🔄 Machine Learning Workflow")

    st.write(
        """
        **Data → Cleaning → Feature Encoding → "
        "Random Forest → Churn Probability → Risk Level**
        """
    )


# ============================================================
# PREDICT CHURN
# ============================================================

elif page == "🔮 Predict Churn":

    st.title("🔮 Predict Customer Churn")

    st.write(
        "Enter customer information to estimate "
        "their probability of churn."
    )

    st.divider()


    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Customer Profile")

    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            df["Gender"].unique()
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            df["Senior Citizen"].unique()
        )


    with col2:

        partner = st.selectbox(
            "Partner",
            df["Partner"].unique()
        )

        dependents = st.selectbox(
            "Dependents",
            df["Dependents"].unique()
        )


    with col3:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )


    # --------------------------------------------------------
    # SERVICES
    # --------------------------------------------------------

    st.subheader("📱 Services")

    col1, col2, col3 = st.columns(3)


    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            df["Phone Service"].unique()
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            df["Multiple Lines"].unique()
        )


    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            df["Internet Service"].unique()
        )

        online_security = st.selectbox(
            "Online Security",
            df["Online Security"].unique()
        )


    with col3:

        online_backup = st.selectbox(
            "Online Backup",
            df["Online Backup"].unique()
        )

        device_protection = st.selectbox(
            "Device Protection",
            df["Device Protection"].unique()
        )


    # --------------------------------------------------------
    # ADDITIONAL SERVICES
    # --------------------------------------------------------

    st.subheader("🛠️ Additional Services")

    col1, col2, col3 = st.columns(3)


    with col1:

        tech_support = st.selectbox(
            "Tech Support",
            df["Tech Support"].unique()
        )


    with col2:

        streaming_tv = st.selectbox(
            "Streaming TV",
            df["Streaming TV"].unique()
        )


    with col3:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            df["Streaming Movies"].unique()
        )


    # --------------------------------------------------------
    # CONTRACT & BILLING
    # --------------------------------------------------------

    st.subheader("💳 Contract & Billing")

    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(
            "Contract",
            df["Contract"].unique()
        )


    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            df["Paperless Billing"].unique()
        )


    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            df["Payment Method"].unique()
        )


    col1, col2 = st.columns(2)


    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=10.0
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Churn Risk",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        customer = {

            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges
        }


        input_df = pd.DataFrame(
            [customer]
        )


        # Encode categorical columns

        for col in categorical_columns:

            if col in input_df.columns:

                input_df[col] = encoders[col].transform(
                    input_df[col].astype(str)
                )


        # Match model feature order

        input_df = input_df[
            feature_columns
        ]


        # Make prediction

        prediction = model.predict(
            input_df
        )[0]


        probability = model.predict_proba(
            input_df
        )[0][1]


        probability_percentage = (
            probability * 100
        )


        # Risk classification

        if probability_percentage >= 70:

            risk = "HIGH RISK"

        elif probability_percentage >= 40:

            risk = "MEDIUM RISK"

        else:

            risk = "LOW RISK"


        # ====================================================
        # CLEAN RESULT
        # ====================================================

        st.divider()

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.error(
                "🔴 " + risk
            )

        else:

            st.success(
                "🟢 " + risk
            )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability_percentage:.1f}%"
            )


        with col2:

            st.metric(
                "Prediction",
                "Likely to Churn"
                if prediction == 1
                else
                "Unlikely to Churn"
            )


        st.write(
            "**Churn Probability**"
        )

        st.progress(
            probability
        )


        if prediction == 1:

            st.warning(
                "⚠️ This customer may require "
                "proactive retention strategies."
            )

        else:

            st.info(
                "ℹ️ This customer currently shows "
                "a lower likelihood of churn."
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Customer Analytics")

    st.write(
        "Explore the customer dataset."
    )

    st.divider()


    total_customers = len(df)

    churned = int(
        df["Churn Value"].sum()
    )

    retained = (
        total_customers - churned
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "Churned",
            f"{churned:,}"
        )


    with col3:

        st.metric(
            "Retained",
            f"{retained:,}"
        )


    st.divider()


    st.subheader(
        "Customer Churn Distribution"
    )


    churn_chart = pd.DataFrame(
        {
            "Customer Status": [
                "Retained",
                "Churned"
            ],

            "Customers": [
                retained,
                churned
            ]
        }
    )


    st.bar_chart(
        churn_chart.set_index(
            "Customer Status"
        )
    )


    st.subheader(
        "Contract Distribution"
    )


    contract_counts = (
        df["Contract"]
        .value_counts()
    )


    st.bar_chart(
        contract_counts
    )


    st.subheader(
        "Internet Service Distribution"
    )


    internet_counts = (
        df["Internet Service"]
        .value_counts()
    )


    st.bar_chart(
        internet_counts
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.subheader(
        "Customer Churn Prediction"
    )

    st.write(
        """
        This project uses machine learning to predict
        whether a telecom customer is likely to churn.

        **Model:** Random Forest Classifier

        **Technologies:**
        Python, Pandas, NumPy, Scikit-learn,
        Streamlit and Joblib.

        **Application Workflow:**

        Customer Information  
        ↓  
        Data Preprocessing  
        ↓  
        Feature Encoding  
        ↓  
        Random Forest Model  
        ↓  
        Churn Probability  
        ↓  
        Risk Classification
        """
    )

    st.divider()

    st.caption(
        "Customer Churn Prediction • Machine Learning Project"
    )