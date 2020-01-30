#pip freeze > requirements.txt

import os
import streamlit as st 
from PIL import Image
from datetime import date, time
import base64

# EDA Pkgs
import pandas as pd 

# Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 

def main():
	""" Common ML Dataset Explorer """
	st.title("Projeto Integrador")
	
	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Grupo 100Vies</p></div>
	"""
	
	st.markdown(html_temp,unsafe_allow_html=True)
	
	page = st.sidebar.selectbox("Escolha uma página", ["Data Scientist","Business"])
	
	if page == "Data Scientist":
		st.header("Explore aqui o seu Dataset")
		visualize_data()
	elif page == "Business":
		st.title("Calculando a probabilidade do sócio sair")
		predicao()

def visualize_data():

	def file_selector(folder_path='./Datasets'):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox("Select A file",filenames)
		return os.path.join(folder_path,selected_filename)

	filename = file_selector()
	st.info("You Selected {}".format(filename))

	# Read Data
	df = pd.read_csv(filename)
	# Show Dataset

	if st.checkbox("Show Dataset"):
		number = st.number_input("Number of Rows to View")
		st.dataframe(df.head(number))

	# Show Columns
	if st.button("Column Names"):
		st.write(df.columns)

	# Show Shape
	if st.checkbox("Shape of Dataset"):
		data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
		if data_dim == 'Rows':
			st.text("Number of Rows")
			st.write(df.shape[0])
		elif data_dim == 'Columns':
			st.text("Number of Columns")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Select Columns To Show"):
		all_columns = df.columns.tolist()
		selected_columns = st.multiselect("Select",all_columns)
		new_df = df[selected_columns]
		st.dataframe(new_df)
	
	# Show Values
	if st.button("Value Counts"):
		st.text("Value Counts By Target/Class")
		st.write(df.iloc[:,-1].value_counts())


	# Show Datatypes
	if st.button("Data Types"):
		st.write(df.dtypes)



	# Show Summary
	if st.checkbox("Summary"):
		st.write(df.describe().T)

	## Plot and Visualization

	st.subheader("Data Visualization")
	# Correlation
	# Seaborn Plot
	if st.checkbox("Correlation Plot[Seaborn]"):
		st.write(sns.heatmap(df.corr(),annot=True))
		st.pyplot()

	
	# Pie Chart
	if st.checkbox("Pie Plot"):
		all_columns_names = df.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.success("Generating A Pie Plot")
			st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

	# Count Plot
	if st.checkbox("Plot of Value Counts"):
		st.text("Value Counts By Target")
		all_columns_names = df.columns.tolist()
		primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
		selected_columns_names = st.multiselect("Select Columns",all_columns_names)
		if st.button("Plot"):
			st.text("Generate Plot")
			if selected_columns_names:
				vc_plot = df.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	# Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
	selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_plot == 'area':
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)

		elif type_of_plot == 'bar':
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)

		elif type_of_plot == 'line':
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)

		# Custom Plot 
		elif type_of_plot:
			cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

def predicao():
	df=pd.read_csv('./Datasets/SOCIOS_USO_CONCATL.csv', encoding='latin-1', delimiter = ';')


	#st.sidebar.slider('slider',1,10)

	#st.multiselect('multiselect', [1,2,3])

	#st.sidebar.selectbox('selectbox',[1,2,3,4])
	meses_assoc = st.sidebar.number_input('Qnt Meses Associação', value=1, min_value = 0, max_value = 1000, step=1)

	pagos_dia = st.sidebar.number_input('Qnt Boletos Pagos em Dia', value=1, min_value = 0, max_value = 1000, step=1)

	pago_atraso = st.sidebar.number_input('Qnt Boletos Pagos Atrasados', value=1, min_value = 0, max_value = 1000, step=1)

	abertos = st.sidebar.number_input('Qnt Boletos Abertos', value=1, min_value = 0, max_value = 1000, step=1)

	cargo1 = st.sidebar.number_input('Qnt Participações - Cargo 1', value=1, min_value = 0, max_value = 1000, step=1)

	cargo2 = st.sidebar.number_input('Qnt Participações - Cargo 2', value=1, min_value = 0, max_value = 1000, step=1)

	cargo3 = st.sidebar.number_input('Qnt Participações - Cargo 3', value=1, min_value = 0, max_value = 1000, step=1)

	total_part = st.sidebar.number_input('Qnt Total de Participações', value=1, min_value = 0, max_value = 1000, step=1)

	meses_sem_part = st.sidebar.number_input('Qnt Meses sem Participações', value=1, min_value = 0, max_value = 1000, step=1)


	#st.sidebar.date_input("date_input", date(2020, 1, 20))

	#st.sidebar.checkbox('checkbox')

	st.write(df)

	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode() 
	href = f'<a href="data:file/csv;base64,{b64}" download="export.csv">Exportar df</a>'
	st.markdown(href, unsafe_allow_html=True)

	import numpy as np
	#image = Image.open('img.png')
	#st.image(image)
	#X_new = np.array(meses_assoc, pagos_dia, pago_atraso, abertos, cargo1, cargo2, cargo3, total_part, meses_sem_part)
	X_new = ([[meses_assoc, pagos_dia, pago_atraso, abertos, cargo1, cargo2, cargo3, total_part, meses_sem_part]])
	X =  pd.DataFrame(df.drop(["COD_EMPRESA","COD_FILIAL","B_PEDIDO_DEMISSAO","B_SOCIO","GRUPO1","GRUPO2","GRUPO3","GRUPO4","GRUPO5","GRUPO6","GRUPO7","GRUPO8","GRUPO9","GRUPO10","GRUPO11","PORTE","PAGO_EM_DIA_POSTERGADO"],axis = 1),)
	y = df.B_PEDIDO_DEMISSAO

	from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

	from sklearn import tree
	# instanciando e ajustando o modelo

	#clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=9,min_samples_split=50)
	#clf = clf.fit(X,y)

	clf_2 = GradientBoostingClassifier(learning_rate= 0.1, max_depth= 6, n_estimators= 50)
	clf_2 = clf_2.fit(X,y)



	#predictions_DT = clf.predict_proba(X_new)
	predictions_GB = clf_2.predict_proba(X_new)


	#st.write(predictions_DT)
	st.write("PROBABILIDADE DO SOCIO SAIR ",round(predictions_GB[0][1]*100,2),'%')
	#st.write(round(predictions_GB[0][1],2),'%')
	#st.write(predictions_GB)
	#st.write(X_new)
	import shap  # package used to calculate Shap values

	# Create object that can calculate shap values
	explainer = shap.TreeExplainer(rfc)

	# calculate shap values. This is what we will plot.
	# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
	shap_values = explainer.shap_values(X)

	# Make plot. Index of [1] is explained in text below.
	shap.summary_plot(shap_values[1], X)
	shap.summary_plot(shap_values, X, plot_type='bar')

if __name__ == '__main__':
	main()