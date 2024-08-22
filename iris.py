import streamlit as st
import pandas as pd
import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X, y)

joblib.dump(dt, 'dt_model.pkl') 

st.write("# IRIS Prediction")

col1, col2, col3, col4 = st.columns(4)

bl = col1.number_input("請輸入花瓣長:")
bw = col1.number_input("請輸入花瓣寬:")
rl = col3.number_input("請輸入花萼長:")
rw = col3.number_input("請輸入花萼寬:")

df_pred = pd.DataFrame([[bl,bw,rl,rw]])

# 加载模型并测试
with open('iris_prediction/dt_model.pkl', 'rb') as f:
    model = joblib.load(f)

prediction = model.predict(df_pred)
prediction_prob = model.predict_proba(df_pred)

if st.button('Predict'):
    if(prediction[0]==0):
        st.write('<p class="big-font">This flower is <font color="#800040">setosa</font>.</p>',unsafe_allow_html=True)
    elif(prediction[0]==1):
        st.write('<p class="big-font">This flower is <font color="red">versicolor</font>.</p>',unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">This flower is <font color="red">virginica</font>.</p>',unsafe_allow_html=True)
        
    st.write('<p class="big-font"><font color="blue">'+str(prediction_prob)+'</font></p>',unsafe_allow_html=True)    
        