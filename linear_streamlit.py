# Importing the necessary libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
loaded_model = pickle.load(open('trained_reg_model.pkl', 'rb'))
import joblib

st.write("""
# Predicting a Penguin's Body Mass
""")
st.write('---')
X = pd.read_csv('penguins_cleaned.csv')

# Streamlit function to collect user input
def user_input_features():
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', float(X['bill_length_mm'].min()), float(X['bill_length_mm'].max()), float(X['bill_length_mm'].mean()))
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', float(X['bill_depth_mm'].min()), float(X['bill_depth_mm'].max()), float(X['bill_depth_mm'].mean()))
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', float(X['flipper_length_mm'].min()), float(X['flipper_length_mm'].max()), float(X['flipper_length_mm'].mean()))

    # Categorical feature selection using radio buttons
    species = st.sidebar.radio('Species', ['Adelie', 'Chinstrap', 'Gentoo'])
    island = st.sidebar.radio('Island', ['Biscoe', 'Dream', 'Torgersen'])
    sex = st.sidebar.radio('Sex', ['Female', 'Male'])

    # Encoding categorical variables based on dummy encoding
    species_Chinstrap = 1 if species == 'Chinstrap' else 0
    species_Gentoo = 1 if species == 'Gentoo' else 0
    island_Dream = 1 if island == 'Dream' else 0
    island_Torgersen = 1 if island == 'Torgersen' else 0
    sex_male = 1 if sex == 'Male' else 0  # Female is the reference category (0)

    # Creating the user input dataframe
    data = {
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'species_Chinstrap': species_Chinstrap,
        'species_Gentoo': species_Gentoo,
        'island_Dream': island_Dream,
        'island_Torgersen': island_Torgersen,
        'sex_male': sex_male
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Collect user input
df = user_input_features()
st.header('Specified Input Parameters')
st.write(df)


# Scale user input
scaler = joblib.load('linear_regression_scaler.pkl')
df_scaled = scaler.transform(df.values.reshape(1, -1))

# Predict using the model
prediction = loaded_model.predict(df_scaled)

# Display prediction
st.header('Prediction of Body Mass (Grams)')
st.write(f"Predicted Body Mass: **{prediction[0]:.2f} grams**")
st.write('---')

# Evaluate the model
#y_pred = model.predict(X_test)
#MAE = mean_absolute_error(y_test, y_pred)
#MSE = mean_squared_error(y_test, y_pred)
#RMSE = np.sqrt(MSE)

#st.write(f"**Mean Absolute Error (MAE):** {MAE:.2f}")
#st.write(f"**Mean Squared Error (MSE):** {MSE:.2f}")
#st.write(f"**Root Mean Squared Error (RMSE):** {RMSE:.2f}")
