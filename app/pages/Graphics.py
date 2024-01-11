import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data\card_transdata.csv')

st.title("Датасет card_transdata")

st.header("Тепловая карта с корреляцией между признаками")

plt.figure(figsize=(19, 9))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта с корреляцией')
st.pyplot(plt)

st.header("Гистограммы")

plt.figure(figsize=(8, 6))
sns.histplot(df.sample(5000)["distance_from_home"], bins=100)
plt.title(f'Гистограмма для distance_from_home')
st.pyplot(plt)

plt.figure(figsize=(8, 6))
sns.histplot(df.sample(5000)["ratio_to_median_purchase_price"], bins=100)
plt.title(f'Гистограмма для ratio_to_median_purchase_price')
st.pyplot(plt)

plt.figure(figsize=(8, 6))
sns.histplot(df.sample(5000)["online_order"], bins=2)
plt.title(f'Гистограмма для online_order')
st.pyplot(plt)

st.header("Ящики с усами ")

columns = ['distance_from_home','ratio_to_median_purchase_price','distance_from_last_transaction']

outlier = df
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]

for col in columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data_filtered[col])
    plt.title(f'{col}')
    plt.xlabel('Значение')
    st.pyplot(plt)

st.header("Круговая диаграмма целевого признака")
plt.figure(figsize=(8, 8))
df['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('fraud')
plt.ylabel('')
st.pyplot(plt)