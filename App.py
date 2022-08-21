import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_player import st_player
from pipeline import input_func, helper_func
import streamlit.components.v1 as components
#transformer = joblib.load('transformer.sav')
#pca = joblib.load('pca.sav')
model = joblib.load('model.sav')
sc=joblib.load('scaler.sav')
oe=joblib.load('onehot.sav')
st.set_page_config(layout="wide")
with st.sidebar:
    
    choose = option_menu("Welcome", ["Home", "Tech Stack","Predictor", "Contributors"],
                         icons=['house', 'stack', 'robot', 'people-fill'],
                         menu_icon="cart", default_index=0, 
                         
                         styles={
                            "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                            "icon": {"color": "White", "font-size": "25px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4d4d4d"},
                            "nav-link-selected": {"background-color": "#4d4d4d"},
                        }
    ) 


with open("contributors.html",'r') as f:
   contributors=f.read();
def html():
    components.html(
      contributors
     ,
    height=1400,
    
    scrolling=True,
)
def pred():
    st.title("Customer Response Predictor")
    
    
    Age=st.number_input('Age',max_value=100)
    Education=st.selectbox("Education",['Basic','Graduation',  'Master', 'PhD', '2n Cycle'])
    x={'Graduation':3, 'PhD':5, 'Master':4, 'Basic':2, '2n Cycle':1}
    Education=x[Education]
    Marital_Status=st.selectbox('Marital Status',['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone','Absurd', 'YOLO'])
    Income=st.slider('Income',max_value=1000000)
    Children=st.slider('Children',max_value=4)
    Recency=st.slider('When was the last time you visited the store?',min_value=0,max_value=100)
    np.random.seed(np.random.randint(1,100))
    MntWines=np.random.randint(0,1000)
    MntFruits=np.random.randint(0,1000)
    MntMeatProducts=np.random.randint(0,600)
    MntFishProducts=np.random.randint(0,150)
    MntSweetProducts=np.random.randint(0,110)
    MntGoldProds=np.random.randint(0,150)
    NumDealsPurchases=np.random.randint(0,15)
    NumWebPurchases=np.random.randint(0,15)
    NumCatalogPurchases=np.random.randint(0,15)
    NumStorePurchases=np.random.randint(0,15)
    NumWebVisitsMonth=np.random.randint(0,20)
    Z_CostContact=np.random.randint(0,3)
    Z_Revenue=np.random.randint(0,15)
    if Income>150000:
        Income=15000
    Complain=st.selectbox('Do Customer has any complain?',['Yes','No'])
    if Complain=='Yes':
        Complain=1
    else:
        Complain=0
    arr= [Age, Education, Income, Recency, MntWines, MntFruits,
       MntMeatProducts, MntFishProducts, MntSweetProducts,
       MntGoldProds, NumDealsPurchases, NumWebPurchases,
       NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
       Complain, Z_CostContact, Z_Revenue] 
    a=oe.transform([[Marital_Status]])
    #st.write(a.toarray())
    a=a.toarray()
    for i in a[0]:
        arr.append(i)
    
    arr.append(Children)
    X=sc.transform([arr])
    if(st.button("Submit")):
        answer=model.predict(X)
        if answer==1:
            st.success('Customer is Intrested in Product.')
        else:
            st.error('Customer is not intrested in Product.')
with open('techstack.html','r') as f:
  techstack=f.read();
def tech():
    components.html(
    techstack
    ,
    height=1000,
    
    scrolling=True,
    )



if choose=="Predictor":

    pred()
elif choose=="Home":
    st.title('Customer Response Predictor')
    st.markdown("<p style='text-align: justify;'>The objective of the project is to diagnostically predict whether or not a customer will resond to the advertisement.\nThis predictor is built for Ecommerce companies and offline stores which uses advertisement. This predictor use the value of Age, Education, Marital status, Number of children, Income and some random factors as input.The accuracy of prediction is 92%.</p>", unsafe_allow_html=True)

    # st.markdown("<h1 style='text-align: center;'>Healthcare AI</h1>", unsafe_allow_html=True)

    with open("pic.html",'r') as f:
        pic=f.read();
    components.html(pic, height=400)

    # def load_lottieurl(url: str):
    #     r = requests.get(url)
    #     if r.status_code != 200:
    #         return None
    #     return r.json()
 
    # lt_url_hello = "https://assets6.lottiefiles.com/packages/lf20_1yy002na.json"
    # lottie_hello = load_lottieurl(lt_url_hello)
 
    # st_lottie(
    #         lottie_hello,  
    #         key="hello",
    #         speed=1,
    #         reverse=False,
    #         loop=True,
    #         quality="low",
    #         height=400,
    #         width=400            
    # )

    
elif choose=="Tech Stack":
    tech()
elif choose=="Contributors":
    html()
