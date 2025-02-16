# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Travel Insurance Claim Predictor')
st.text('This web can be used to predict persons travel insurance claim')



# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    
    # Numerical Features
    duration = st.sidebar.slider('Duration', min_value=0, max_value=365, value=30)
    net_sales = st.sidebar.number_input('Net Sales', min_value=-357.5, max_value=682.0, value=50.0)
    commission = st.sidebar.number_input('Commision (in value)', min_value=0.0, max_value=262.76, value=10.0)
    age = st.sidebar.slider('Age', min_value=0, max_value=88, value=35)

    # Categorical Features
    agency = st.sidebar.selectbox('Agency', ['C2B', 'JZI', 'EPX', 'CWT', 'LWC', 'ART', 'CSR', 'RAB', 'KML', 
                                             'SSI', 'TST', 'ADM', 'CCR', 'CBH', 'TTW'])
    agency_type = st.sidebar.radio('Agency Type', ['Airlines', 'Travel Agency'])
    distribution_channel = st.sidebar.radio('Distribution Channel', ['Online', 'Offline'])
    
    product_name = st.sidebar.selectbox('Product Name', [
        'Annual Silver Plan', 'Basic Plan', '2 way Comprehensive Plan', 'Bronze Plan', 'Cancellation Plan', 
        '1 way Comprehensive Plan', 'Rental Vehicle Excess Insurance', 'Single Trip Travel Protect Gold', 
        'Silver Plan', 'Value Plan', '24 Protect', 'Comprehensive Plan', 'Ticket Protector', 
        'Travel Cruise Protect', 'Gold Plan', 'Single Trip Travel Protect Silver', 'Premier Plan', 
        'Single Trip Travel Protect Platinum', 'Annual Gold Plan', 'Annual Travel Protect Gold', 
        'Annual Travel Protect Silver', 'Individual Comprehensive Plan', 'Annual Travel Protect Platinum', 
        'Travel Cruise Protect Family', 'Child Comprehensive Plan', 'Spouse or Parents Comprehensive Plan'
    ])
    
    destination = st.sidebar.selectbox('Destination', [
        'SINGAPORE', 'INDIA', 'UNITED STATES', 'KOREA, REPUBLIC OF', 'THAILAND', 'JAPAN', 'INDONESIA', 
        'MALAYSIA', 'VIET NAM', 'AUSTRALIA', 'FINLAND', 'UNITED KINGDOM', 'SRI LANKA', 'SPAIN', 'HONG KONG', 
        'MACAO', 'CHINA', 'UNITED ARAB EMIRATES', 'IRAN, ISLAMIC REPUBLIC OF', 'TAIWAN, PROVINCE OF CHINA', 
        'CANADA', 'PHILIPPINES', 'BELGIUM', 'TURKEY', 'BRUNEI DARUSSALAM', 'DENMARK', 'SWITZERLAND', 
        'NETHERLANDS', 'SWEDEN', 'KENYA', 'MYANMAR', 'FRANCE', 'GERMANY', 'RUSSIAN FEDERATION', 'ARGENTINA', 
        'POLAND', 'TANZANIA, UNITED REPUBLIC OF', 'SERBIA', 'ITALY', 'CROATIA', 'NEW ZEALAND', 'PERU', 
        'MONGOLIA', 'CAMBODIA', 'GREECE', 'QATAR', 'CZECH REPUBLIC', 'NORWAY', 'LUXEMBOURG', 'MALTA', 
        'PAKISTAN', 'ISRAEL', 'SAUDI ARABIA', 'AUSTRIA', 'PORTUGAL', 'NEPAL', 'UKRAINE', 'ESTONIA', 'ICELAND', 
        'BRAZIL', 'MEXICO', "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'CAYMAN ISLANDS', 'PANAMA', 'TUNISIA', 
        'IRELAND', 'ETHIOPIA', 'NORTHERN MARIANA ISLANDS', 'MALDIVES', 'SOUTH AFRICA', 'VENEZUELA', 
        'BANGLADESH', 'OMAN', 'JORDAN', 'MALI', 'CYPRUS', 'MAURITIUS', 'KUWAIT', 'AZERBAIJAN', 'BAHRAIN', 
        'HUNGARY', 'BHUTAN', 'BELARUS', 'MOROCCO', 'ECUADOR', 'UZBEKISTAN', 'KAZAKHSTAN', 'LEBANON', 'CHILE', 
        'FIJI', 'PAPUA NEW GUINEA', 'FRENCH POLYNESIA', 'NIGERIA', 'GEORGIA', 'SLOVENIA', 'COLOMBIA', 
        'ZIMBABWE', 'NAMIBIA', 'BULGARIA', 'BERMUDA', 'URUGUAY', 'GUINEA', 'VANUATU', 'EGYPT', 'GHANA', 'GUAM', 
        'UGANDA', 'BOLIVIA', 'JAMAICA', 'LATVIA', 'REPUBLIC OF MONTENEGRO', 'KYRGYZSTAN', 'GUADELOUPE', 
        'ZAMBIA', 'RWANDA', 'BOTSWANA', 'ROMANIA', 'GUYANA', 'LITHUANIA', 'GUINEA-BISSAU', 'COSTA RICA', 
        'SENEGAL', 'CAMEROON', 'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF', 'SAMOA', 'PUERTO RICO', 
        'TAJIKISTAN', 'ARMENIA', 'DOMINICAN REPUBLIC', 'MOLDOVA, REPUBLIC OF', 'REUNION'
    ])

    # Creating a dictionary with user input
    user_data = {
        'Duration': duration,
        'Net Sales': net_sales,
        'Commision (in value)': commission,
        'Age': age,
        'Agency': agency,
        'Agency Type': agency_type,
        'Distribution Channel': distribution_channel,
        'Product Name': product_name,
        'Destination': destination
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df


# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'Model Final Travel Insurance.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('This customer will Claim')
    else:
        st.write('This customer will not Claim')
    
    # Displaying the probability of the customer buying
    st.write(f"Probability of Claim: {probability[1]:.2f}")  # Probability of class 1 (BUY)
