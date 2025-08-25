import pickle

import streamlit as st
import pandas as pd
from os import path
import numpy as np
# import joblib

#st.title("Hello world")

#st.write("Good to see you")

#creating a dataframe
#df_Data = pd.DataFrame({'col1': [1, 2, 3,5], 'col2': ['a', 'b', 'c', 'd']})

#st.write(df_Data) #displaying the dataframe we created


#st.title("Iris dataset")
#df_Iris = pd.read_csv(path.join("Data", "iris.csv"))
#st.write(df_Iris)
#filepath = Root/Data/iris.csv

#st.scatter_chart(df_Iris[['sepal_length', 'sepal_width']])

#df_map = pd.DataFrame(np.array([[8.69263923698824, 76.77689597066033]]),
                       #columns=["lat", "lon"])
#st.map(df_map)

#petal_length = st.number_input("Please choose a petal length",min_value=3,max_value=5)

st.title("Flower Species predictor")
petal_length=st.number_input("please choose the petal length",
                             placeholder="please enter the petal length",
                             min_value=1.0, max_value=6.9,value=None)
petal_width = st.number_input("Please choose a petal width", placeholder="please enter the petal length",
                             min_value=0.1, max_value=2.5,value=None)
sepal_length = st.number_input("Please choose a sepal length",  placeholder="please enter the petal length",
                               min_value=4.3, max_value=7.9,value=None)
sepal_width = st.number_input("Please choose a sepal width", placeholder="please enter the petal length",
                              min_value=1.0, max_value=6.9,value=None)

#iris_predictor = path.join("Model","iris_classifier.pkl")
user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width,]],
                          columns=['sepal_length','sepal_width','petal_length','petal_width',])
# st.write(user_input)

model_path =path.join("Model","iris_classifier.pkl")
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)

dict_species = {0:'Setosa',1:'Versicolor',2:'Virginica'}


if st.button("Predict Species"):
    if ((petal_length==None) or (petal_width==None)
            or (sepal_length==None) or (sepal_width==None)):
        st.write("please fill the value")

    else:
        # predicted can be done
        predicted_species = iris_predictor.predict(user_input)
        st.write("the species is", predicted_species)

