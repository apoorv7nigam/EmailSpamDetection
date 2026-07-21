import pandas as pd
import joblib
import streamlit as st
import re

# ML resources ko cache kiya taaki deployment ke baad app lag na kare
@st.cache_resource
def load_ml_resources():
    try:
        model = joblib.load("email_spam_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: 'email_spam_model.pkl' or 'scaler.pkl' not found.")
        return None, None

def main():
    st.set_page_config(
        page_title="Email Spam Detection", 
        page_icon="📧",
        layout="centered"
    )

    # 🎨 CUSTOM STYLING (Fixed Error: Changed unsafe_allow_index to unsafe_allow_html)
    st.markdown("""
        <style>
        /* Main application background padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Heading gradient effect */
        .main-title {
            font-size: 42px !important;
            font-weight: 800 !important;
            background: linear-gradient(45deg, #FF4B4B, #4A90E2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        
        /* Input card structural styling */
        div[data-testid="stContainer"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
        }
        
        /* Target dark/light mode compatibility for texts inside container */
        div[data-testid="stContainer"] p, div[data-testid="stContainer"] h3 {
            color: #212529 !important;
        }

        /* Predict button layout adjustments */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(74, 144, 226, 0.4);
            background: linear-gradient(135deg, #357ABD 0%, #286090 100%) !important;
        }
        </style>
    """, unsafe_allow_html=True)  # <--- Yahan error fix ho gaya hai!

    # Gradient App Header
    st.markdown('<h1 class="main-title">📧 Email Spam Detection</h1>', unsafe_allow_html=True)
    st.write("Predict whether an email is Spam or Not Spam using Machine Learning.")
    st.markdown("---")

    # Resume Side-panel: Recruiters ko project ka quick overview dene ke liye
    with st.sidebar:
        st.header("About Project")
        st.markdown("""
        This project uses a Machine Learning model to classify emails based on their metadata features.
        
        **Key Steps:**
        * Feature Extraction
        * Data Scaling
        * Classification Prediction
        """)

    model, scaler = load_ml_resources()
    if model is None or scaler is None:
        return

    # Inputs Container
    with st.container():
        st.markdown("### 📊 Input Features")
        
        col1, col2 = st.columns(2)

        with col1:
            subject = st.selectbox(
                "Select Subject",
                ("Security Alert", "Win Prize", "Invoice", "Meeting", "Offer", "Project Update", "Greetings", "Account Verification")
            )
            
            sender_email = st.text_input("Sender Email",placeholder="example@gmail.com")
            
            attachment = st.selectbox("Has Attachment",("No","Yes"))
            
            email_body = st.text_area("Paste Complete Email",height=250,
                                      placeholder="Paste complete email here...")

            # Show warning if email is longer than training range
            if len(email_body.strip()) > 265:
                st.warning(
                    f"⚠️ Your email contains {len(email_body.strip())} characters.\n\n"
                    "This model was trained on emails between 20 and 265 characters. "
                    "For the most reliable prediction, please use emails within this range."
                )

    # Mappings
    subject_map = {"Security Alert": 0, "Win Prize": 1, "Invoice": 2, "Meeting": 3, "Offer": 4, "Project Update": 5, "Greetings": 6, "Account Verification": 7}
    domain_map = {"outlook.com": 0, "yahoo.com": 1, "gmail.com": 2, "company.com": 3, "Unknown": 4}
    
    p1 = subject_map[subject]
    p6 = 0 if attachment=="No" else 1
    email_length = len(email_body)
    num_links = len(
        re.findall(r'https?://\S+|www\.\S+', email_body)
    )
    num_special = len(
        re.findall(r'[^A-Za-z0-9\s]', email_body)
    )
    capital_words = len(
        re.findall(r'\b[A-Z]{2,}\b', email_body)
    )
    if "@" in sender_email:
        domain = sender_email.split("@")[-1].lower()
    else:
        domain = "Unknown"

    # domain_map = {
    #     "outlook.com":0,
    #     "yahoo.com":1,
    #     "gmail.com":2,
    #     "company.com":3
    # }

    p7 = domain_map.get(domain,4)

    # Exact DataFrame mapping
    data_new = pd.DataFrame({
        "Subject": [p1],
        "Email_Length": [email_length],
        "Num_Links": [num_links],
        "Num_Special_Chars": [num_special],
        "Capital_Words": [capital_words],
        "Has_Attachment": [p6],
        "Email_Domain": [p7]
    })

    st.markdown(" ") # Spacing

    # Styled Custom Predict Button
    if st.button("Predict", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            st.subheader("📊 Extracted Features")
            feature_df = pd.DataFrame({
                "Feature":[
                    "Email Length",
                    "Links",
                    "Special Characters",
                    "Capital Words",
                    "Email Domain"
                ],
                "Value":[
                    email_length,
                    num_links,
                    num_special,
                    capital_words,
                    domain
                ]
            })
            st.dataframe(feature_df, use_container_width=True)
            data_scaled = scaler.transform(data_new)
            prediction = model.predict(data_scaled)[0]
            probability = model.predict_proba(data_scaled)[0]

        st.markdown("### 🎯 Prediction Result")
        
        # Result panel display
        if prediction == 1:
            st.error(f"### 🚨 This Email is SPAM (Risk: {probability[1]*100:.2f}%)")
            st.markdown("""
            **Analysis Warning:** The structural distribution of features matches patterns typically found in phishing layouts or unsolicited automated broadcast metrics.
            """)
        else:
            st.success(f"### ✅ This Email is NOT SPAM (Safe Score: {probability[0]*100:.2f}%)")
            st.markdown("""
            **Analysis Report:** The input profile is well within standard deviation boundaries for normal operational or personal emails.
            """)

if __name__ == "__main__":
    main()