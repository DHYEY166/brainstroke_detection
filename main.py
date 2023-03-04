import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


header = st.container()
description = st.container()
dataset = st.container()
model_training = st.container()

with description:
	st.header("PROJECT DESCRIPTION")
	st.text("""Here we used several prediction models such as Logistic Regression Model, 
Decision Tree Model, SVM model, and Naive Bayes Model for the predection of 
Brain-Stroke on a dataset that we obtained from Kaggle.""")


with header:
	st.title(" :book: BRAIN STROKE PREDECTION USING VARIOUS MACHINE LEARNING MODELS")
	st.text('In this project I have developed a Brain Stroke Prediction System')


with dataset:
	st.header('Brain-Stroke Detection Dataset')
	st.text("""      I found this dataset on kaggle.com""")
	st.text("""The stroke prediction dataset was used to perform the study. 
There were 4981 rows and 10 columns in this dataset. 
The value of the output column(stroke) is either 1 or 0. 
The number 0 indicates that there no risk of a stroke to occur, 
while the value 1 indicates that there is a risk that a stroke might occur.""")
	df = pd.read_csv('data/full_data.csv')
	st.write(df.head(20))

	#st.text('PREDICTION DISTRIBUTION WHERE "0" REPRESENTS "NO STROKE" AND "1" REPRESENTS "Stroke"')
	
	#p_dist = pd.DataFrame(df['stroke'].value_counts()).head(50)
	#st.bar_chart(p_dist)

#with features:
#	st.header('Select Features')
 #   feature_cols = st.multiselect('Select the columns to use as input features', df.columns.tolist())


with model_training:

	st.header("MODEL TRAINING")
	df = pd.read_csv('data/full_data.csv')
	st.text("""Here you get to choose the hyperparameters of the model 
and see how the performance of a model changes.""")

	sel_col, disp_col = st.columns(2)
	C = sel_col.slider('Select regularization strength (smaller values for stronger regularization)', 0.01, 10.0, 1.0, 0.01)


	age_outliers=df.loc[df['age']>70]
	age_outliers['bmi'].shape
	df["age"] = df["age"].apply(lambda x: 70 if x>570 else x)
	df["age"] = df["age"].fillna(28.4)
	cat_df = df[['gender','Residence_type','smoking_status','stroke']]
	summary = pd.concat([pd.crosstab(cat_df[x], cat_df.stroke) for x in cat_df.columns[:-1]], keys=cat_df.columns[:-1])
	df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
	df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)
	df["gender"] = df["gender"].apply(lambda x: 1 if x=="Male" else 0)
	df = pd.get_dummies(data=df, columns=['smoking_status'])
	df = pd.get_dummies(data=df, columns=['work_type'])
	std=StandardScaler()
	columns = ['avg_glucose_level','bmi','age']
	scaled = std.fit_transform(df[['avg_glucose_level','bmi','age']])
	scaled = pd.DataFrame(scaled,columns=columns)
	df=df.drop(columns=columns,axis=1)
	df=df.merge(scaled, left_index=True, right_index=True, how = "left")
	df0 = df.drop(['stroke'], axis=1)
	feature_cols = sel_col.multiselect('Select the columns to use as input features', df0.columns.tolist())

	

	X = df[feature_cols].values 
	y = df['stroke'].values


	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.svm import SVC
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


	models = {
	    'Logistic Regression': LogisticRegression(),
	    'Decision Tree': DecisionTreeClassifier(),
	    'Random Forest': RandomForestClassifier(),
	    'Support Vector Machine': SVC(kernel='sigmoid', gamma='scale'),
	    'Naive Bayes': GaussianNB()
	}

	model_name = sel_col.selectbox('Select a machine learning model', list(models.keys()))


	def train_and_evaluate_model(X, y, model_name):
		model = models[model_name]
		model.fit(X, y)
		y_pred = model.predict(X)
		accuracy = accuracy_score(y, y_pred)
		cm = confusion_matrix(y, y_pred)
		return accuracy, cm

	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
	accuracy, cm = train_and_evaluate_model(X, y, model_name)

	disp_col.write(accuracy)
	disp_col.write(cm)





















