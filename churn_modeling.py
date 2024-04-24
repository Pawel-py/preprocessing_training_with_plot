import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

url = 'https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/kaggle+/churn_modelling/Telco-Customer-Churn.csv'
df = pd.read_csv(url)
df.head()
df.info()

#%%


total_charges_median = df[df['TotalCharges'] != ' ']['TotalCharges'].astype('float').median()
df['TotalCharges'][df['TotalCharges'] == ' '] = total_charges_median
df['TotalCharges'] = df[df['TotalCharges'] != ' ']['TotalCharges'].astype('float')

df['TotalCharges'].value_counts()

#%%
if not df.isnull().sum().sum():
    print('Nie ma braków')

#%%
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'Contract', 'StreamingMovies',
                   'PaperlessBilling', 'PaymentMethod', 'Churn']
     
numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in categorical:
    df[col] = pd.Categorical(df[col])
for num in numerical:
    df[num] = df[num].astype('float')

df.info()

#%%
df = df.drop(columns = ['customerID'])
df.info()
#%%
df.describe()
df.describe(include = ['category'])

#%% churn pie plot 
plt.figure(figsize=[8,6])
sns.set()
df['Churn'].value_counts().plot(kind='pie', 
                                fontsize = 16, 
                                colors = ['green', 'red'],
                                explode=[0.1, 0],
                                shadow = True,
                                autopct = '%1.1f%%')
plt.legend()
plt.ylabel(' ')
plt.title('Rozkład Churn')
#%% gender pie plot
plt.figure(figsize=[8,6])
sns.set()
df['gender'].value_counts().plot(kind='pie', 
                                fontsize = 16,
                                colors = ['green', 'red'],
                                explode=[0.1, 0],
                                shadow = True,
                                autopct = '%1.1f%%')
plt.legend(['Male','Female'])
plt.ylabel(' ')
plt.title('Rozkład Płci')
#%%
plt.figure(figsize=[8,6])
sns.set()
df['SeniorCitizen'].value_counts().plot(kind='pie', 
                                fontsize = 16, 
                                colors = ['green', 'red'],
                                explode=[0.1, 0],
                                shadow = True,
                                autopct = '%1.1f%%')
plt.legend()
plt.ylabel(' ')
plt.title('Rozkład zmiennej SeniorCitizen')
#%% uniwersalna funkcja
def pie_plot(x):
    plt.figure(figsize=[8,6])
    sns.set()
    df[x].value_counts().plot(kind='pie', 
                                    fontsize = 16, 
                                    colors = ['green', 'red'],
                                    explode=[0.1, 0],
                                    shadow = True,
                                    autopct = '%1.1f%%')
    plt.legend()
    plt.ylabel(' ')
    plt.title('Rozkład zmiennej {}'.format(x))

