# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:20:11 2024

@author: Ketan
"""

import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import numpy as np

diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_trained_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_trained_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction Using ML')
    #getting the inputs from the user
    col1, col2, col3=st.columns(3)
    with col1:
        Pregnancies=st.text_input('Number of Pregnancies')
    with col2:
        Glucose=st.text_input('Glucose level')
    with col3:
        BloodPressure=st.text_input('BloodPressure value')
    with col1:
        SkinThickness=st.text_input('SkinThickness value')
    with col2:
        Insulin=st.text_input('Insulin value')
    with col3:
        BMI=st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age=st.text_input('Age')    
        
    #code for prediction
    diab_diagnosis=''
    if(st.button('Diabetes Test Result')):
        diab_pred=diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(diab_pred[0]==0):
            diab_diagnosis='The person is not diabetic'
        else:
            diab_diagnosis='The person is diabetic'
    st.success(diab_diagnosis)

if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction Using ML')
    # getting the inputs from the user

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        age = int(age) if age else None
    with col2:
        sex = st.text_input('Sex')
        sex = int(sex) if sex else None
    with col3:
        cp = st.text_input('Chest pain type')
        cp = int(cp) if cp else None
    with col1:
        trestbps = st.text_input('Resting blood pressure')
        trestbps = int(trestbps) if trestbps else None
    with col2:
        chol = st.text_input('Serum cholesterol in mg/dL')
        chol = int(chol) if chol else None
    with col3:
        fbs = st.text_input('Fasting blood sugar > 120 mg/dL')
        fbs = int(fbs) if fbs else None
    with col1:
        restecg = st.text_input('Resting electrocardiographic results')
        restecg = int(restecg) if restecg else None
    with col2:
        thalach = st.text_input('Maximum heart rate achieved')
        thalach = int(thalach) if thalach else None
    with col3:
        exang = st.text_input('Exercise induced angina')
        exang = int(exang) if exang else None
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise relative to rest')
        oldpeak = float(oldpeak) if oldpeak else None
    with col2:
        slope = st.text_input('The slope of the peak exercise ST segment')
        slope = int(slope) if slope else None
    with col3:
        ca = st.text_input('Number of major vessels (0-3) colored by fluoroscopy')
        ca = int(ca) if ca else None
    with col1:
        thal = st.text_input('0 = normal; 1 = fixed defect; 2 = reversible defect')
        thal = int(thal) if thal else None

    # code for prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        heart_pred = heart_disease_model.predict(input_data)
        if heart_pred[0] == 0:
            heart_diagnosis = 'The person does not have Heart Disease'
        else:
            heart_diagnosis = 'The person has Heart Disease'
    st.success(heart_diagnosis)

    

if selected == 'Parkinsons Prediction':
    # page title
    st.title('Parkinsons Disease Prediction Using ML')
    
    col1, col2, col3, col4=st.columns(4)
    with col1:
        name=st.text_input('ASCII subject name and recording number')
    with col2:
        fo=st.text_input('  Average vocal fundamental frequency : Fo(Hz)')
    with col3:
        fhi=st.text_input("Maximum vocal fundamental frequency: Fhi(Hz)")
    with col4:
        flo=st.text_input('Minimum vocal fundamental frequency: Flo(Hz)')
    with col1:
        jitter=st.text_input('Jitter(%)')
    with col2:
        jitter_abs=st.text_input('Jitter(Abs)')
    with col3:
        rap=st.text_input('RAP')
    with col4:
        prq=st.text_input('PPQ')
    with col1:
        ddp=st.text_input('DDP')
    with col2:
        shimmer=st.text_input('Shimmer')
    with col3:
        shimmer_db=st.text_input('Shimmer(dB)')
    with col4:
        apq3=st.text_input('APQ3')
    with col1:
        apq5=st.text_input('APQ5')
    with col2:
        apq=st.text_input('APQ')
    with col3:
       dda=st.text_input('DDA')
    with col4:
        nhr=st.text_input('NHR')
    with col1:
        hnr=st.text_input('HNR')
    with col2:
        rpde=st.text_input('RPDE')
    with col3:
        dfa=st.text_input('DFA')
    with col4:
        spread1=st.text_input('spread1')
    with col1:
        spread2=st.text_input('spread2')
    with col2:
        d2=st.text_input('D2')
    with col3:
        ppe=st.text_input('PPE')
        
    #code for prediction
    scaler=StandardScaler()
    parkin_diagnosis=''
    if(st.button('Diabetes Test Result')):
        input_data=[fo,fhi,flo,jitter,jitter_abs,rap,prq,ddp,shimmer,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]
        input_array=np.array(input_data).reshape(1,-1)
        input_res=scaler.fit_transform(input_array)
        parkin_pred=parkinsons_model.predict(input_res)
        if(parkin_pred[0]==0):
            parkin_diagnosis='The person does not have Parkinsons Disease'
        else:
            parkin_diagnosis='The person has Parkinsons Disease'
    st.success(parkin_diagnosis)