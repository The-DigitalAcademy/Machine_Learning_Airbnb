import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import plotly.express as px

st.set_page_config(layout='wide', page_title='Wizards Airbnb', page_icon='🏠')

st.title('Wizards Airbnb')

st.write("~~~~~~~~~")
st.markdown("<h1 style='text-align: center; color: white;'>Wizards Airbnb</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white;'>'🏠</h1>", unsafe_allow_html=True)
st.write("~~~~~~~~~")
st.markdown("<h2 style='text-align: center; color: white;'>Welcome to Wizards Airbnb!</h2>", unsafe_allow_html=True)
st.write("~~~~~~~~~~~")

df = pd.read_csv("airbnb_MOD.csv")

with st.form(key='display options'):
    st.write("1. Display Airbnb Price Prediction\n")
    year = st.selectbox("Choose the year: ", [2016,2017,2018,2019,2020,2021,2022,2023])
    day_of_week = st.selectbox("Choose Which Day: ", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    month_ = st.selectbox("Choose Which Month: ", [1,2,3,4,5,6,7,8,9,10,11,12])
    seasons = st.selectbox("Choose A Season:", [1,2,3,4])
    
    
    # year = st.selectbox("Choose the year: ", df['Year'].unique())
    # day_of_week = st.selectbox("Choose Which Day: ", df['Day_Of_Week'].unique())
    # month_ = st.selectbox("Choose Which Month: ", df['Month'].unique())
    # seasons = st.selectbox("Choose A Season:", df['Seasons'].unique())
    
    type_algorthm = st.selectbox("Choose A Machine Learning Algorthm:", ["Decision Tree", "Random forest", 'Lasso', 'Ridge', 'Linear Regression',' Adaboost Regression', 'Stacking Regression'])
    submit_details = st.form_submit_button('Submit!')
    
    price_val = st.sidebar.slider('Price range:', float(df.price.min()), 10000., (70000., 00.))
    f = px.histogram(df.query(f'price.between{price_val}'), x='price', nbins=15, title='Price distribution')
    f.update_xaxes(title='Price')
    f.update_yaxes(title='No. of listings')
    st.plotly_chart(f)
     
    neighborhood = st.sidebar.radio("Neighborhood Group:", df.neighbourhood_group.unique())
    show_exp = st.sidebar.checkbox("Include expensive listings")
    st.write("Selected Neighborhood Group:", neighborhood)
    st.write("Include Expensive Listings:", show_exp)
    
    room_type = st.sidebar.radio("Room Type:", df.room_type.unique())
    st.write("Selected Room Type:", room_type)
    
    min_nights = st.sidebar.slider('Minimum Nights:', float(df.minimum_nights.min()), 365., (365., 0.))
    filtered_data = df.query(f'minimum_nights.between{min_nights}')
    fig = px.histogram(filtered_data, x='minimum_nights', nbins=15, title='Minimum Nights')
    fig.update_xaxes(title='Minimum Nights')
    fig.update_yaxes(title='No. of Listings')
    st.plotly_chart(fig)
    
    avail_365 = st.sidebar.slider('Availability:', float(df.availability_365.min()), 365., (365., 0.))
    filtered_data = df.query(f'availability_365.between{min_nights}')
    fig = px.histogram(filtered_data, x='availability_365', nbins=15, title='Availability')
    fig.update_xaxes(title='Availability')
    fig.update_yaxes(title='No. of Listings')
    st.plotly_chart(fig)


    if submit_details:
    
        user_input = pd.DataFrame({
            'Year' :[year],
            'Day_Of_Week': [day_of_week],
            'Month' :[ month_],
            'Seasons' :[seasons],
            'price' : [price_val],
            'group' : [neighborhood],
            'room_type' : [room_type],
            'minimum_nights' : [min_nights],
            'availability_365' : [avail_365]  
        })
        
        
        scale = StandardScaler()
        scaledX = scale.fit_transform(user_input)
        # if type_algorthm == "Decision Tree":
        model_load_path = "BnB.pkl"
        with open(model_load_path,'rb') as file:
            model = pickle.load(file)
                        
        
        test_y_pred = model.predict(user_input)
       
        st.write("Predicted Average Price:", test_y_pred[0])
     

st.balloons()
