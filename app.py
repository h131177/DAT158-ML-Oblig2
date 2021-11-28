#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[2]:


from pycaret.regression import load_model, predict_model 
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os


# In[3]:


class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('final_model') 
        self.save_fn = 'path.csv'     
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False)  
    
    def preprocess(self, data):
   # Preprocess method insipred by code from predict.py from the Flask-tutorial hospitalapp

   # This method makes sure all features are defined going into the model
    
        null = None
        
        # Setting default values for some of the features, while letting the models imputation method deal with the other potentially 
        # missing values (that are  not retrieved from user)
        feature_values = {
            'Id':null, 'MSSubClass': null, 'MSZoning': null, 'LotFrontage':null, 'LotArea':null, 'Street': null,
       'Alley': null, 'LotShape': null, 'LandContour':null, 'Utilities':null, 'LotConfig':null,
       'LandSlope':null, 'Neighborhood':null, 'Condition1':null, 'Condition2':null, 'BldgType':null,
       'HouseStyle':null, 'OverallQual':null, 'OverallCond':null, 'YearBuilt':null, 'YearRemodAdd':null,
       'RoofStyle':null, 'RoofMatl':null, 'Exterior1st':null, 'Exterior2nd':null, 'MasVnrType':null,
       'MasVnrArea':0, 'ExterQual':null, 'ExterCond':null, 'Foundation':null, 'BsmtQual':null,
       'BsmtCond':null, 'BsmtExposure':null, 'BsmtFinType1':null, 'BsmtFinSF1':0,
       'BsmtFinType2':null, 'BsmtFinSF2':0, 'BsmtUnfSF':0, 'TotalBsmtSF':0, 'Heating':null,
       'HeatingQC':null, 'CentralAir':null, 'Electrical':null, '1stFlrSF':0, '2ndFlrSF':0,
       'LowQualFinSF':0, 'GrLivArea':null, 'BsmtFullBath':0, 'BsmtHalfBath':0, 'FullBath':1,
       'HalfBath':0, 'BedroomAbvGr':null, 'KitchenAbvGr':1, 'KitchenQual':null,
       'TotRmsAbvGrd':5, 'Functional':null, 'Fireplaces':0, 'FireplaceQu':null, 'GarageType':null,
       'GarageYrBlt':null, 'GarageFinish':null, 'GarageCars':null, 'GarageArea':0, 'GarageQual':null,
       'GarageCond':null, 'PavedDrive':null, 'WoodDeckSF':0, 'OpenPorchSF':0,
       'EnclosedPorch':0, '3SsnPorch':0, 'ScreenPorch':0, 'PoolArea':0, 'PoolQC':null,
       'Fence':null, 'MiscFeature':null, 'MiscVal':0, 'MoSold':null, 'YrSold':null, 'SaleType':null,
       'SaleCondition':null
        }

        # Parse the form inputs and return the features updated with values entered.
        for key in [k for k in data.keys() if k in feature_values.keys()]:
            feature_values[key] = data[key]

        return feature_values
    
    def run(self):
        #image = Image.open('../assets/human-heart.jpg')
        #st.image(image, use_column_width=False)
    
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch')) #bruke batch for aa predikere paa alle bildene. 
        st.sidebar.info('This app is created to predict house price' )
        st.sidebar.success('DAT158')
        st.title('House price prediction')
        
       
        if add_selectbox == 'Online': 
            
            

            
            output=''
            input_dict = self.preprocess(user_input)
            input_df = pd.DataFrame(input_dict, index=[0])
        
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
                output = output['Label'][0].round(2)
                
            
            st.success('Predicted price: {}'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)
            
sa = StreamlitApp()
sa.run()





