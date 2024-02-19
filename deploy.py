# %%
import streamlit as st

# %%
def main():
    st.title('Car insurance cross and up selling signal')

    sex_list = ["Male", "Female"]
    driving_license_list = ['Yes','No']
    
    # Create a selectbox widget with unique keys
    sex_option = st.selectbox("Select customer sex:", sex_list, key='sex_option')
    age_option = st.slider('Select customer age:', min_value=0.0, max_value=99.0, value=25.0, step=1.0, format="%.1f", key='age_slider')
    driving_license_option = st.selectbox("Select customer driving license status:", driving_license_list, key='sex_option')

    tag_string = ''
    if st.button('Tag prediction !!'):
        tag_string = sex_option + str(age_option) + driving_license_option
        
    st.success(tag_string)

# %%
if __name__ == '__main__':
    main()


