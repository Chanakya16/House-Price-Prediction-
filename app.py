import streamlit as st 
from joblib import load
import pandas as pd
from PIL import Image
import numpy as np 



def load_model():

    models = {'Linear_Regression' :load('Models/linearregression.joblib'),
              'Random_Forest_Regressor' : load('Models/randomforestregressor.joblib'),
              'XGBoost' : load('Models/xgboost.joblib',mmap_mode=None),
            }
    
    return models

models = load_model()


def load_metrices(model_name):

    me = f"Reports/{model_name}_me.csv"
    re = f"Reports/{model_name}_re.png"
    me = pd.read_csv(me)
    me = pd.DataFrame(me)
    re = Image.open(re)

    return me,re







def predict_price(model_name, bedrooms, bathrooms, sqft_living, lot_size, year_built, tax_value):

    model = models.get(model_name)

    if not model:
        raise ValueError("Invalid model name")

    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_.tolist()
    else:
        raise ValueError("Model does not have feature names information.")

   
    input_values = {
        'bedroomcnt': float(bedrooms),
        'bathroomcnt': float(bathrooms),
        'calculatedfinishedsquarefeet': float(sqft_living),
        'lotsizesquarefeet': float(lot_size),
        'yearbuilt': float(year_built),
        'taxvaluedollarcnt': float(tax_value)
    }

    
    full_features = {feature: float(input_values.get(feature, 0)) for feature in model_features}

   
    df_input = pd.DataFrame([full_features])

    
    df_input = df_input.astype(float)

   
    df_input = df_input.select_dtypes(include=[np.number])

   
    df_input = df_input.reindex(columns=model_features, fill_value=0.0)

    
    print("DEBUG: Checking for non-numeric values in DataFrame")
    print(df_input.applymap(lambda x: isinstance(x, (int, float))).all())  

    print("DEBUG: DataFrame types after conversion:")
    print(df_input.dtypes)  

    print("DEBUG: Any NaN values?")
    print(df_input.isna().sum())  

    print("DEBUG: Model expects features:", model.feature_names_in_)
    print("DEBUG: Input DataFrame columns:", df_input.columns)

   
    object_cols = df_input.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"ERROR: Found object columns in DataFrame! {object_cols}")
        print("DEBUG: Sample problematic values:")
        print(df_input[object_cols].head())

    try:
        predicted_price = model.predict(df_input)[0]
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return None  

   
    estimated_median_price = 400000  
    y_pred = estimated_median_price * np.exp(predicted_price)

    return y_pred

 


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Price Prediction", "Model Performance"])

    if page == "Price Prediction":
        
        st.title("House Price Prediction")


        st.image("preview (2).webp")



        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
        sqft_living = st.number_input("Square Footage of Living Area", min_value=500, max_value=10000)
        lot_size = st.number_input("Lot Size in Square Feet", min_value=500, max_value=20000)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
        tax_value = st.number_input("Tax Assessed Value", min_value=10000, max_value=10000000)
        options = ['Linear_Regression', 'Random_Forest_Regressor', 'XGBoost']
        model_name = st.selectbox('Choose an option:', options)


        if st.button("Predict Price"):
            user_features = [bedrooms, bathrooms, sqft_living, lot_size, year_built, tax_value] 

            prediction = predict_price(model_name,*user_features)  

            
            st.subheader(f"Predicted House Price: $ {prediction:,.2f}")
    elif page == "Model Performance":
        
        st.title("Model Performance")
        options1 = ['Linear_Regression', 'Random_Forest_Regressor', 'XGBoost']
        model_name = st.selectbox('Choose an option:', options1)
        model_name = model_name.lower()


        if model_name:
            me , re = load_metrices(model_name)
            st.table(me)
            st.image(re, use_container_width=True)



if __name__ == '__main__':
    main()


