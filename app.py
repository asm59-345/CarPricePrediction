import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.set_page_config(layout="wide")
st.title('Car Price Prediction Website')

st.image("https://placehold.co/1200x300/003366/FFFFFF?text=Car+Price+Predictor&font=inter")


@st.cache_data
def load_data():
    cars_data = pd.read_csv('Cardetails.csv')
    cars_data.dropna(inplace=True)
    cars_data.drop_duplicates(inplace=True)
    
    def get_brand_name(car_name):
        car_name = car_name.split(' ')[0]
        return car_name.strip()
    cars_data['name'] = cars_data['name'].apply(get_brand_name)
    
    unique_brands = sorted(cars_data['name'].unique())
    unique_fuels = cars_data['fuel'].unique()
    unique_sellers = cars_data['seller_type'].unique()
    unique_transmissions = cars_data['transmission'].unique()
    unique_owners = cars_data['owner'].unique()
    
    min_year = int(cars_data['year'].min())
    max_year = int(cars_data['year'].max())
    max_kms = int(cars_data['km_driven'].max())
    
    return (unique_brands, unique_fuels, unique_sellers, unique_transmissions, 
            unique_owners, min_year, max_year, max_kms)

(unique_brands, unique_fuels, unique_sellers, unique_transmissions, 
 unique_owners, min_year, max_year, max_kms) = load_data()


if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None

with st.form("prediction_form"):
    st.header("Enter Car Details")

    st.subheader("Vehicle Details")
    name = st.selectbox('Select Car Brand', unique_brands, key="name")
    year = st.slider('Car Manufactured Year', min_year, max_year, max_year - 5, key="year")
    km_driven = st.slider('No of Kms Driven', min_value=11, max_value=max_kms, value=50000, key="km")

    st.subheader("Ownership & Type")
    fuel = st.selectbox('Fuel Type', unique_fuels, key="fuel")
    seller_type = st.selectbox('Seller Type', unique_sellers, key="seller")
    transmission = st.selectbox('Transmission Type', unique_transmissions, key="transmission")
    owner = st.selectbox('Owner Type', unique_owners, key="owner") 

    st.subheader("Engine & Performance")
    mileage = st.slider('Car Mileage (kmpl)', min_value=8, max_value=40, value=20, key="mileage")
    engine = st.slider('Engine CC', min_value=600, max_value=6000, value=1500, key="engine")
    max_power = st.slider('Max Power (bhp)', min_value=30, max_value=600, value=100, key="max_power")
    torque = st.slider('Torque (Nm)', min_value=40, max_value=800, value=200, key="torque")
    seats = st.slider('No of Seats', min_value=2, max_value=12, value=5, key="seats")

    st.markdown("---") 
    submit_button = st.form_submit_button(label='Predict Price', type="primary", use_container_width=True)

if submit_button:
 
    model_columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                     'transmission', 'owner', 'mileage', 'engine', 
                     'max_power', 'torque', 'seats']

    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, 
          owner, mileage, engine, max_power, torque, seats]],
        columns=model_columns)
    
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
           'Fourth & Above Owner', 'Test Drive Car'],
                                  [1,2,3,4,5], inplace=True)
    
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)
    
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)
    
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
    
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
           'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
           'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
           'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
           'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
           inplace=True)

    car_price = model.predict(input_data_model)
    st.session_state.predicted_price = car_price[0]


if st.session_state.predicted_price is not None:
    st.header("Prediction Result")
    st.metric(label="Estimated Car Price", value=f"₹ {st.session_state.predicted_price:,.2f}")
    st.balloons()    