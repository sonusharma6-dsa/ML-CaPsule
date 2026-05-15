# setup all libraries based on requirement for UI
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
# page config
st.set_page_config(page_title="CKD Predictor", page_icon="💊", layout="wide")

# styling
st.markdown("""
<style>
body,.stApp{background:#0f1724;color:#dde6f2}
[data-testid="stSidebar"]{background:#141d2c}
.block {
    background: #1a2434;
    border: 1px solid #2e3d52;
    padding:18px;
    border-radius:10px;
    margin-bottom:15px;           
}
.title {
    color: #8cb8ff;
    font-size: 2rem;
    font-weight: 700;            
}
.stButton button {
    width:100%;
    border:none;
    background: #4285f4;
    color: white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)


# dataset and model
@st.cache_resource
def train_model():
    np.random.seed(7)
    n = 300

    ckd = np.random.choice([0, 1], n, p=[0.4, 0.6])

    df = pd.DataFrame({
        "age": np.where(ckd, np.random.normal(58, 12, n), np.random.normal(40, 10, n)),
        "bp": np.where(ckd, np.random.normal(85, 10, n), np.random.normal(72, 8, n)),
        "sugar": np.where(ckd, np.random.randint(1, 5, n), 0),
        "creatinine": np.where(ckd, np.random.normal(3.8, 1, n), np.random.normal(1, 0.2, n)),
        "hemo": np.where(ckd, np.random.normal(10, 2, n), np.random.normal(14, 1, n)),
        "class": ckd
    })

    x = df.drop("class", axis=1)
    y = df["class"]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=11)

    model = ExtraTreesClassifier()
    model.fit(xtrain, ytrain)

    return model, scaler, df


model, scaler, data = train_model()

# Side Bar for CKD

with st.sidebar:
    st.title("💊 CKD Tools")
    page = st.radio("", ["overview", "Prediction", "Charts"])

# overview

if page == "overview":

    st.markdown("""
    <div class = 'block'>
      <div class = 'title'>Chromic kideny Disease Predictor</div>
      <p>
        This app predicts CKD risk from basic health values
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Features", 5)
    c3.metric("Model", "Extra Trees")

    fig = px.scatter(
        data,
        x="age",
        y="creatinine",
        color=data["class"].map({0: "healthy", 1: "CKD"}),
        title="Creatine vs Age"
    )

    st.plotly_chart(fig, use_container_width=True)

# prediction

elif page == "Prediction":
    st.subheader("Patient Details")

    c1, c2, c3 = st.columns(3)

    age = c1.slider("Age", 1, 90, 45)
    bp = c2.slider("Blood Pressure", 50, 180, 80)
    sugar = c3.selectbox("Suger", [0, 1, 2, 3, 4])

    c4, c5 = st.columns(2)

    creatinine = c4.number_input("Creatinine", 0.1, 15.0, 1.2)
    hemo = c5.number_input("haemoglobin", 3.0, 18.0, 13.5)

    if st.button("Check Risk"):

        row = np.array([[age, bp, sugar, creatinine, hemo]])

        pred = model.predict(scaler.transform(row))[0]
        prob = model.predict_proba(scaler.transform(row))[0][1] * 100

        if pred == 1:
            st.error(f"CKD Risk Detected ({prob:.1f}%)")
        else:
            st.success(f"Low CKD Risk ({100 - prob:.1f}%)")

        st.progress(int(prob))


# charts

else:

    st.subheader("Dataset Analysis")

    chart_data= data.copy()

    chart_data["status"] = chart_data["class"].map({0: "healthy", 1: "CKD"})

    fig1 = px.histogram(
        chart_data,
        x="hemo",
        color="status",
        barmode="overlay",
        title="Haemoglobin Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    imp = pd.DataFrame({
        "Feature": ["age", "bp", "sugar", "creatinine", "hemo"],
        "Score": model.feature_importances_
    })

    fig2 = px.bar(
        imp.sort_values("Score"),
        x="Score",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    st.plotly_chart(fig2, use_container_width=True)

