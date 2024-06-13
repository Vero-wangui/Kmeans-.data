import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
data3= pd.read_csv('data3.csv',encoding='latin1')
print(data3.head())
print(data3.shape)
print(data3.columns)
print(data3.info())
print(data3.describe())
print(data3.isnull().sum())
df_null=round(100*(data3.isnull().sum())/len(data3),2)
print(df_null)
# Droping rows having missing values
data3=data3.drop('StockCode',axis=1)
print(data3.shape)

# Changing Customer Id as str
data3['CustomerID']=data3['CustomerID'].astype(str)
print(data3.info())

#data preparation
data3['Amount']=data3['Quantity']*data3['UnitPrice']
print(data3.info())
data3_monetary =data3.groupby('CustomerID')['Amount'].sum()
print(data3_monetary.head())
# Grouping by Country and calculating total sales
sales_by_country=data3.groupby('Country')['Amount'].sum().sort_values(ascending=False)
print(sales_by_country.head())
# Grouping by Description and calculating total quantity sold
sales_by_product = data3.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(sales_by_product.head())

#frequently bought item
frequency_data=data3.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
print(frequency_data.head())

#total last month sales
last_sales=data3.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
print(last_sales.head())

# Convert to datetime to proper datatype
data3['InvoiceDate'] = pd.to_datetime(data3['InvoiceDate'],format='%d-%m-%Y %H:%M')
print(data3.head())

# Compute the maximum date to know the last transaction date
max_date = max(data3['InvoiceDate'])
print(max_date)

# Compute the minimum date to know the last transaction date
min_date = min(data3['InvoiceDate'])
print(min_date)

#compute the diiference
time_diff=(max_date-min_date)
print(time_diff)

last_month=(max_date-pd.DateOffset(months=1)).month
last_month_year=(max_date-pd.DateOffset(months=1)).year
last_month_sales=data3[(data3['InvoiceDate'].dt.month==last_month)&(data3['InvoiceDate'].dt.year==last_month_year)
]
print('last month sales data:')
print(last_month_sales)

# Calculate the total sales amount for the last 30 days
totalsales = last_month_sales['Quantity']*last_month_sales['UnitPrice']
totalsales=totalsales.sum()
print(f'total sales for last month: {totalsales}')
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

df2=data3.groupby("Description").agg({"Quantity":"sum","UnitPrice":"sum"}).reset_index()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_3=scaler.fit_transform(df2[["Quantity","UnitPrice"]])

df2.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters = 3,random_state =0,n_init='auto')
kmeans.fit(df_3)
df2["Clusters"]= kmeans.predict(df_3)

from sklearn.metrics import silhouette_score
perf=silhouette_score(df_3,kmeans.labels_,metric="euclidean")
print(perf)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df2[['Quantity']],df2[['UnitPrice']],test_size=0.3,random_state=0)

from sklearn import preprocessing
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

'''Testing a number of clusters to determine how many to use'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    '''Train the model for the current value of k on the training model'''
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(df_3)
    fit.append(model)
    score.append(silhouette_score(df_3,model.labels_,metric='euclidean'))
print(fit)
print(score)

sns.lineplot(x=K,y=score)
plt.show()



