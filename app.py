import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main layout */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headings */
    h1 {
        font-size: 2.3rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }

    /* Metrics */
    /* ---------- METRIC CARDS ---------- */

[data-testid="stMetric"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 1.1rem !important;
}

/* Metric label */
[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* Metric number */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 750 !important;
}

/* Metric number's inner elements */
[data-testid="stMetricValue"] div {
    color: #ffffff !important;
}

/* Metric delta, if present */
[data-testid="stMetricDelta"] {
    color: #cbd5e1 !important;
}

    /* Buttons */
    .stButton > button {
        min-height: 48px;
        border-radius: 10px;
        font-weight: 650;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    /* Number inputs */
    div[data-baseweb="input"] > div {
        border-radius: 9px;
    }

    /* Progress bar */
    div[data-testid="stProgress"] {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Remove excessive spacing */
    hr {
        margin: 1.5rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/Telco_churn.csv")


model = load_model()
df = load_data()

df.columns = df.columns.str.strip()


# ============================================================
# DATA PREPROCESSING
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

for column in drop_columns:
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)


# Convert Total Charges to numeric

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

categorical_columns = [
    column
    for column in df.columns
    if df[column].dtype == "object"
]


encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    encoder.fit(
        df[column].astype(str)
    )

    encoders[column] = encoder


# ============================================================
# MODEL FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column != "Churn Value"
]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@st.cache_data
def calculate_model_metrics():

    X = df[feature_columns].copy()
    y = df["Churn Value"]

    for column in categorical_columns:

        X[column] = encoders[column].transform(
            X[column].astype(str)
        )

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    matrix = confusion_matrix(
        y,
        predictions
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        matrix
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📡 Telco Intelligence")

    st.caption(
        "Customer Churn Prediction"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔮 Predict Churn",
            "📊 Analytics",
            "🧠 Model Performance",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption("Machine Learning")

    st.write("🌲 Random Forest Classifier")

    st.caption(
        "End-to-end customer churn analytics"
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("📡 Telco Churn Intelligence")

    st.write(
        "A machine learning dashboard for identifying "
        "customers at risk of churn and supporting "
        "proactive retention decisions."
    )

    st.divider()


    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    total_customers = len(df)

    churned_customers = int(
        df["Churn Value"].sum()
    )

    retained_customers = (
        total_customers -
        churned_customers
    )

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100


    st.subheader("Dataset Overview")


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
            "Retained Customers",
            f"{retained_customers:,}"
        )


    with col4:

        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.1f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # PROJECT PURPOSE
    # --------------------------------------------------------

    st.subheader("Turning Customer Data into Retention Insights")


    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.markdown("### 🎯 Project Purpose")

            st.write(
                "Customer churn is a major challenge for "
                "telecommunication businesses. The objective "
                "of this project is to identify customers who "
                "may be likely to leave the service."
            )

            st.write(
                "The model uses customer demographics, "
                "services, contract information and billing "
                "details to estimate churn risk."
            )


    with col2:

        with st.container(border=True):

            st.markdown("### 💡 Business Value")

            st.write(
                "Churn predictions can help businesses "
                "identify high-risk customers before they leave."
            )

            st.write(
                "These insights can support targeted retention "
                "campaigns, personalized offers and proactive "
                "customer engagement."
            )


    st.divider()


    # --------------------------------------------------------
    # MACHINE LEARNING WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 Machine Learning Workflow")


    workflow = [
        ("01", "Customer Data"),
        ("02", "Data Cleaning"),
        ("03", "Feature Encoding"),
        ("04", "Random Forest"),
        ("05", "Risk Prediction")
    ]


    cols = st.columns(5)


    for col, (number, name) in zip(cols, workflow):

        with col:

            with st.container(border=True):

                st.markdown(f"### {number}")

                st.write(name)


    st.divider()


    # --------------------------------------------------------
    # QUICK INSIGHT
    # --------------------------------------------------------

    st.subheader("📌 Key Dataset Insight")


    st.info(
        f"Out of {total_customers:,} customers, "
        f"{churned_customers:,} have churned, resulting in "
        f"an overall churn rate of {churn_rate:.1f}%."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Predict Churn":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter the customer's profile, service and billing "
        "information to estimate their churn risk."
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


    st.divider()


    # --------------------------------------------------------
    # SERVICE INFORMATION
    # --------------------------------------------------------

    st.subheader("📱 Service Information")


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


    st.divider()


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


    st.divider()


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
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Analyze Customer Risk",
        type="primary",
        use_container_width=True
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # ENCODE CATEGORICAL VARIABLES
        # ----------------------------------------------------

        for column in categorical_columns:

            if column in input_df.columns:

                input_df[column] = encoders[
                    column
                ].transform(
                    input_df[column].astype(str)
                )


        # Match model feature order

        input_df = input_df[
            feature_columns
        ]


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        probability = model.predict_proba(
            input_df
        )[0][1]


        probability_percentage = (
            probability * 100
        )


        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

        if probability_percentage >= 70:

            risk = "High Risk"

            recommendation = (
                "This customer shows a high likelihood "
                "of churn. Consider proactive retention "
                "actions such as personalized offers, "
                "service reviews or targeted outreach."
            )

            status = "🔴"


        elif probability_percentage >= 40:

            risk = "Medium Risk"

            recommendation = (
                "This customer shows a moderate likelihood "
                "of churn. Consider monitoring engagement "
                "and introducing suitable retention initiatives."
            )

            status = "🟠"


        else:

            risk = "Low Risk"

            recommendation = (
                "This customer currently shows a lower "
                "likelihood of churn. Continue maintaining "
                "a positive customer experience."
            )

            status = "🟢"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 Churn Risk Assessment")


        # Risk status

        if prediction == 1:

            st.error(
                f"{status} {risk}"
            )

        else:

            st.success(
                f"{status} {risk}"
            )


        # Main metrics

        col1, col2, col3 = st.columns(3)


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


        with col3:

            st.metric(
                "Risk Level",
                risk
            )


        # Probability visualization

        st.write("**Estimated Churn Probability**")

        st.progress(
            probability
        )


        # Recommendation

        st.subheader("💡 Recommended Action")

        st.info(
            recommendation
        )


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Customer Analytics")

    st.write(
        "Explore customer churn patterns across contracts, "
        "services, payment methods and tenure."
    )

    st.divider()


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    total_customers = len(df)

    churned = int(
        df["Churn Value"].sum()
    )

    retained = (
        total_customers -
        churned
    )

    churn_rate = (
        churned /
        total_customers
    ) * 100


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "Churned Customers",
            f"{churned:,}"
        )


    with col3:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Customer Churn Distribution")


    churn_distribution = pd.DataFrame(
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
        churn_distribution.set_index(
            "Customer Status"
        )
    )


    st.divider()


    # --------------------------------------------------------
    # CONTRACT ANALYSIS
    # --------------------------------------------------------

    st.subheader("Churn by Contract Type")


    contract_churn = (
        df.groupby("Contract")[
            "Churn Value"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )


    st.bar_chart(
        contract_churn
    )


    st.divider()


    # --------------------------------------------------------
    # INTERNET SERVICE
    # --------------------------------------------------------

    st.subheader("Churn by Internet Service")


    internet_churn = (
        df.groupby("Internet Service")[
            "Churn Value"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )


    st.bar_chart(
        internet_churn
    )


    st.divider()


    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    st.subheader("Churn by Payment Method")


    payment_churn = (
        df.groupby("Payment Method")[
            "Churn Value"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )


    st.bar_chart(
        payment_churn
    )


    st.divider()


    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    st.subheader("Churn by Customer Tenure")


    tenure_bins = pd.cut(
        df["Tenure Months"],
        bins=[
            -1,
            6,
            12,
            24,
            48,
            72,
            120
        ],
        labels=[
            "0–6 months",
            "7–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months",
            "73+ months"
        ]
    )


    tenure_churn = (
        df.groupby(
            tenure_bins,
            observed=False
        )["Churn Value"]
        .mean()
        * 100
    )


    st.bar_chart(
        tenure_churn
    )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "🧠 Model Performance":

    st.title("🧠 Model Performance")

    st.write(
        "Evaluation metrics for the trained "
        "Random Forest classification model."
    )

    st.divider()


    (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        matrix
    ) = calculate_model_metrics()


    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.1f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.1f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.1f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.1f}%"
        )


    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )


    st.divider()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")


    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual: Retained",
            "Actual: Churned"
        ],
        columns=[
            "Predicted: Retained",
            "Predicted: Churned"
        ]
    )


    st.dataframe(
        matrix_df,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # METRIC INTERPRETATION
    # --------------------------------------------------------

    st.subheader("📌 Understanding the Metrics")


    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.markdown("### Accuracy")

            st.write(
                "The percentage of customers correctly "
                "classified by the model."
            )

            st.markdown("### Precision")

            st.write(
                "Among customers predicted to churn, "
                "the proportion who actually churned."
            )


    with col2:

        with st.container(border=True):

            st.markdown("### Recall")

            st.write(
                "Among customers who actually churned, "
                "the proportion correctly identified."
            )

            st.markdown("### F1 Score")

            st.write(
                "A balance between precision and recall."
            )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")


    st.write(
        "Customer Churn Intelligence is an end-to-end "
        "machine learning application designed to predict "
        "customer churn in the telecommunications industry."
    )


    st.divider()


    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")


    st.write(
        "The objective is to identify customers who are "
        "more likely to leave a telecom service and provide "
        "risk-based insights that can support customer "
        "retention strategies."
    )


    st.divider()


    # --------------------------------------------------------
    # ML APPROACH
    # --------------------------------------------------------

    st.subheader("🧠 Machine Learning Approach")


    st.write(
        "A Random Forest Classifier is used to estimate "
        "the probability of customer churn based on "
        "demographic, service, contract and billing features."
    )


    st.write(
        "Categorical variables are encoded using LabelEncoder "
        "before being passed to the trained model."
    )


    st.divider()


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 Application Workflow")


    workflow = [
        "Customer Data",
        "Data Cleaning",
        "Feature Encoding",
        "Random Forest Model",
        "Churn Probability",
        "Risk Classification",
        "Retention Recommendation"
    ]


    for index, step in enumerate(workflow, start=1):

        st.write(
            f"**{index}.** {step}"
        )


    st.divider()


    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader("🛠️ Technology Stack")


    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "Streamlit",
                "Joblib"
            ],

            "Purpose": [
                "Programming language",
                "Data processing",
                "Numerical computation",
                "Machine learning",
                "Interactive web application",
                "Model serialization"
            ]
        }
    )


    st.dataframe(
        technologies,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    st.caption(
        "Telco Customer Churn Intelligence • "
        "Machine Learning Portfolio Project"
    )