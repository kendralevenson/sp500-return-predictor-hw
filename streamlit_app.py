import streamlit as st
import boto3
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io, json

ENDPOINT_NAME = "kpca-lasso-pipeline-endpoint-v1"
BUCKET_NAME   = "franck-soh-s3-bucket"
EXPLAINER_KEY = "explainer/explainer_pca.shap"
REGION        = "us-east-1"

st.set_page_config(page_title="SP500 Return Predictor", layout="wide")
st.title("SP500 Cumulative Return Predictor - Option 1")
st.markdown("Upload SP500Data.csv and get GOOGL cumulative return predictions.")

st.sidebar.header("AWS Credentials")
aws_access_key    = st.sidebar.text_input("Access Key ID", type="password")
aws_secret_key    = st.sidebar.text_input("Secret Access Key", type="password")
aws_session_token = st.sidebar.text_input("Session Token (optional)", type="password")

@st.cache_resource
def get_boto_clients(access_key, secret_key, session_token, region):
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token or None,
        region_name=region,
    )
    return session.client("sagemaker-runtime"), session.client("s3")

def invoke_endpoint(runtime_client, endpoint_name, data_array):
    buf = io.StringIO()
    pd.DataFrame(data_array).to_csv(buf, header=False, index=False)
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=buf.getvalue(),
    )
    return np.array(json.loads(response["Body"].read().decode())).flatten()

@st.cache_resource
def load_shap_explainer(_s3_client):
    obj = _s3_client.get_object(Bucket=BUCKET_NAME, Key=EXPLAINER_KEY)
    return shap.Explainer.load(io.BytesIO(obj["Body"].read()))

st.subheader("1. Upload Data")
uploaded = st.file_uploader("Upload SP500Data.csv", type="csv")

if uploaded:
    df = pd.read_csv(uploaded, index_col=0)
    st.write(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head())
    return_period = st.slider("Return period (days)", 1, 20, 5)
    if st.button("Predict"):
        if not aws_access_key or not aws_secret_key:
            st.error("Enter AWS credentials in the sidebar.")
        else:
            with st.spinner("Calling SageMaker endpoint..."):
                try:
                    rc, s3c = get_boto_clients(aws_access_key, aws_secret_key, aws_session_token, REGION)
                    drop_cols = [c for c in ["GOOGL","MSFT","AAPL"] if c in df.columns]
                    X = np.log(df.drop(drop_cols, axis=1)).diff(return_period)
                    X = np.exp(X).cumsum()
                    X.columns = [n + "_CR_Cum" for n in X.columns]
                    X_clean = X.fillna(0)
                    preds = invoke_endpoint(rc, ENDPOINT_NAME, X_clean.values)
                    result_df = pd.DataFrame({"Date": X.index, "Predicted_GOOGL_FR_Cum": preds})
                    st.success("Predictions received!")
                    st.dataframe(result_df)
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(result_df["Date"], result_df["Predicted_GOOGL_FR_Cum"])
                    ax.set_title("Predicted GOOGL Cumulative Future Return")
                    plt.xticks(rotation=45); plt.tight_layout()
                    st.pyplot(fig)
                    st.subheader("2. SHAP Explanation")
                    try:
                        exp = load_shap_explainer(s3c)
                        sv  = exp(X_clean.values[:50])
                        shap.plots.waterfall(sv[0], show=False)
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.warning(f"SHAP unavailable: {e}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload SP500Data.csv to get started.")
