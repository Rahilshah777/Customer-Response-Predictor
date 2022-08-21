import streamlit as st
def pred():
    st.title("Diabetese Predictor for Women")
    Pregnancies = st.slider("Number Of Pregnancies", 0, 13 )
    if Pregnancies>=0:
        st.write('Pregnancies :',Pregnancies)
    Glucose =st.slider("Glucose Level", 0, 200)
    if Glucose>=0:
        st.write('Glucose :',Glucose)
    BloodPressure=st.slider("Blood Pressure Level", 70, 150)
    if BloodPressure>=0:
        st.write('BloodPressure :',BloodPressure)
    SkinThickness=st.slider("Blood Pressure Level", 0, 70)
    if SkinThickness>=0:
        st.write('SkinThickness :',SkinThickness)
    Insulin=st.slider("Insulin Level",0, 70)
    if Insulin>=0:
        st.write('Insulin :',Insulin)
    BMI=st.slider("BMI", 15, 40)
    if BMI>=0:
        st.write('BMI :',BMI)
    Age=st.slider("Age", 18, 100)
    if Age>=0:
        st.write('Age :',Age)
pred()