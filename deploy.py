# %%
import warnings
warnings.simplefilter("ignore")

# %%
import streamlit as st

# %%
import pandas as pd

# %%
import pickle

# %% [markdown]
# # Import trained model

# %%
region_code_encoder = pickle.load(open('encoder model pickle/region_code_encoder.pickle', 'rb'))

# %%
vehicle_age_encoder = pickle.load(open('encoder model pickle/vehicle_age_encoder.pickle', 'rb'))

# %%
policy_sales_channel_encoder = pickle.load(open('encoder model pickle/policy_sales_channel_encoder.pickle', 'rb'))

# %%
logistic_rus = pickle.load(open('prediction model pickle/logistic_rus_prediction.pickle', 'rb'))

# %% [markdown]
# # Create function

# %%
def convert_gender(df):
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    return df

# %%
def convert_driving_license(df):
    df['Driving_License'] = df['Driving_License'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

# %%
def convert_previously_insured(df):
    df['Previously_Insured'] = df['Previously_Insured'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

# %%
def convert_vehicle_damage(df):
    df['Vehicle_Damage'] = df['Vehicle_Damage'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

# %%
def encode_region_code(df, encoder):
    encoded = encoder.transform(df[['Region_Code']]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Region_Code']))
    df = df.join(encoded_df).drop('Region_Code', axis=1)
    return df

# %%
def encode_vehicle_age(df, encoder):
    encoded = encoder.transform(df[['Vehicle_Age']]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Vehicle_Age']))
    df = df.join(encoded_df).drop('Vehicle_Age', axis=1)
    return df

# %%
def encode_policy_sales_channel(df, encoder):
    encoded = encoder.transform(df[['Policy_Sales_Channel']]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Policy_Sales_Channel']))
    df = df.join(encoded_df).drop('Policy_Sales_Channel', axis=1)
    return df

# %% [markdown]
# # Prediction function

# %%
def prediction_model(df):

    result = ''

    df = convert_gender(df)
    df = convert_driving_license(df)
    df = convert_previously_insured(df)
    df = convert_vehicle_damage(df)
    df = encode_region_code(df, region_code_encoder)
    df = encode_vehicle_age(df, vehicle_age_encoder)
    df = encode_policy_sales_channel(df, policy_sales_channel_encoder)

    df['predictions'] = logistic_rus.predict(df)

    if df['predictions'][0] == 1:
        result = 'Likely to buy'
    else:
        result = 'Not likely to buy'

    return result

# %%
def main():
    st.title('Car insurance selling signal')

    sex_list = ['Male', 'Female']
    driving_license_list = ['Yes','No']
    region_code_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19',
                        '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37',
                        '38','39','40','41','42','43','44','45','46','47','48','49','50','51','52']
    previously_insured_list = ['Yes','No']
    vehicle_age_list = ['> 2 Years', '1-2 Year', '< 1 Year']
    vehicle_damage_list = ['Yes','No']
    policy_sales_channel_list = ['1','2','3','4','6','7','8','9','10','11','12','13','14','15','16','17','18','19',
                                 '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35',
                                 '36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52',
                                 '53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69',
                                 '70','71','73','74','75','76','78','79','80','81','82','83','84','86','87','88','89',
                                 '90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105',
                                 '106','107','108','109','110','111','112','113','114','115','116','117','118','119',
                                 '120','121','122','123','124','125','126','127','128','129','130','131','132','133',
                                 '134','135','136','137','138','139','140','143','144','145','146','147','148','149',
                                 '150','151','152','153','154','155','156','157','158','159','160','163']
    
    sex_option = st.selectbox("Customer sex:", sex_list, key='sex_option')
    age_option = st.slider('Customer age:', min_value=0, max_value=100, value=25, step=1, key='age_option')
    driving_license_option = st.selectbox("Customer driving license status:", driving_license_list, key='driving_license_option')
    region_code_option = st.selectbox("Customer region code:", region_code_list, key='region_code_option')
    previously_insured_option = st.selectbox("Customer previously insured:", previously_insured_list, key='previously_insured_option')
    vehicle_age_option = st.selectbox("Customer vehicle age:", vehicle_age_list, key='vehicle_age_option')
    vehicle_damage_option = st.selectbox("Customer vehicle damage:", vehicle_damage_list, key='vehicle_damage_option')
    annual_premium_option = st.number_input('Customer annual premium:', min_value=0, value=0)
    policy_sales_channel_option = st.selectbox("Customer policy sales channel:", policy_sales_channel_list, key='policy_sales_channel_option')
    vintage_option = st.slider('Customer vintage:', min_value=0, max_value=500, value=100, step=1, key='vintage_option')
    
    df_prediction = pd.DataFrame({
        'Gender': [sex_option],
        'Age': [age_option],
        'Driving_License': [driving_license_option],
        'Region_Code': [region_code_option],
        'Previously_Insured': [previously_insured_option],
        'Vehicle_Age': [vehicle_age_option],
        'Vehicle_Damage': [vehicle_damage_option],
        'Annual_Premium': [annual_premium_option],
        'Policy_Sales_Channel': [policy_sales_channel_option],
        'Vintage': [vintage_option]
        })
    
    prediction_result = ''

    if st.button('Predict'):
        prediction_result = prediction_model(df_prediction)
    st.success(f'Prediction Result: {prediction_result}')

# %%
if __name__ == '__main__':
    main()


