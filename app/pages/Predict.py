import streamlit as st
import pandas as pd
import pickle
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, rand_score
from tensorflow.keras.models import load_model

uploaded_file = st.file_uploader("Выберите файл датасета")


with open('models\GaussianNB.pkl', 'rb') as file:
    gnb = pickle.load(file)
with open('models\BaggClass.pkl', 'rb') as file:
    bagging_model = pickle.load(file)
with open('models\GBM_classifier.pkl', 'rb') as file:
    gradient_model = pickle.load(file)
with open('models\StClasf.pkl', 'rb') as file:
    stacking_model= pickle.load(file)
with open('models\Kmeans.pkl', 'rb') as file:
    kmeans = pickle.load(file)
tf_model = load_model('models\TF.h5')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Загруженный датасет:", df)
    
    X=df.drop(['fraud'],axis=1)
    y=df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    predictions_gnb = gnb.predict(X_test)
    predictions_bagging_model = bagging_model.predict(X_test)
    predictions_gradient_model = gradient_model.predict(X_test)
    predictions_stacking_model = stacking_model.predict(X_test)
    predictions_kmeans = kmeans.predict(X_test)
    probabilities_tf = np.around(tf_model.predict(X_test))

    # Оценить результаты
    accuracy_gnb = accuracy_score(y_test, predictions_gnb)
    rand_score_kmeans = rand_score(y_test, predictions_kmeans)
    accuracy_gradient_model = accuracy_score(y_test, predictions_gradient_model)
    accuracy_stacking_model = accuracy_score(y_test, predictions_stacking_model)
    accuracy_predictions_bagging_model = accuracy_score(y_test, predictions_bagging_model)
    accuracy_tf = accuracy_score(y_test, probabilities_tf)

    st.success(f"Точность Stacking Classifier: {accuracy_stacking_model}")
    st.success(f"Точность GaussianNB: {accuracy_gnb}")
    st.success(f"Rand Score Kmeans: {rand_score_kmeans}")
    st.success(f"Точность Gradient: {accuracy_gradient_model}")
    st.success(f"Точность Bagging Classifier: {accuracy_predictions_bagging_model}")
    st.success(f"Точность Tensorflow: {accuracy_tf}")


st.title("Получить прогноз о мошенничестве.")

st.header("distance_from_home:")
st.write("Расстояние между местом транзакции по банковской карте и домом (метры).")
distance_from_home = st.number_input("Число:", value=0.17)

st.header("distance_from_last_transaction:")
st.write("Расстояние между местом транзакции по банковской карте и последней транзакции (метры).")
distance_from_last_transaction = st.number_input("Число:", value=0.56)

st.header("ratio_to_median_purchase_price:")
st.write("Отношение самой последней транзакции к средней цене предыдущих транзакций.")
ratio_to_median_purchase_price = st.number_input("Число:", value=3.4)

st.header("repeat_retailer:")
st.write("Произошла ли транзакция у одного и того же продавца.")
repeat_retailer = int(st.toggle("Число:", value=1, key = "repeat_retailer"))

st.header("used_chip")
st.write("Был ли использован аналог csv, для США - CHIP.")
used_chip = int(st.toggle("Число:", value=0, key = "used_chip"))

st.header("used_pin_number")
st.write("Был ли использован пин код.")
used_pin_number = int(st.toggle("Число:", value=0, key = "used_pin_number"))

st.header("online_order:")
st.write("Была ли покупка совершена онлайн.")
online_order = int(st.toggle("online_order", value=1, key = "online_order"))

data = pd.DataFrame({'distance_from_home': [distance_from_home],
                    'distance_from_last_transaction': [distance_from_last_transaction],
                    'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                    'repeat_retailer': [repeat_retailer],
                    'used_chip': [used_chip],
                    'used_pin_number': [used_pin_number],
                    'online_order': [online_order],          
                    })


button_clicked = st.button("Предсказать")

if button_clicked:

    st.header("GaussianNB:")
    pred =[]
    knn_pred = int(gnb.predict(data)[0])
    pred.append(knn_pred)
    st.write(f"{knn_pred}")

    st.header("Bagging:")
    bagging_pred = int(bagging_model.predict(data)[0])
    pred.append(bagging_pred)
    st.write(f"{bagging_pred}")

    st.header("Gradient:")
    gradient_pred = int(gradient_model.predict(data))
    pred.append(gradient_pred)
    st.write(f"{gradient_pred}")

    st.header("Stacking:")
    stacking_pred = int(stacking_model.predict(data)[0])
    pred.append(stacking_pred)
    st.write(f"{stacking_pred}")

    st.header("Perceptron:")
    tf_pred = int(np.round(tf_model.predict(data)[0][0]))
    pred.append(tf_pred)
    st.write(f"{tf_pred}")

    st.header("Финальный результат:")
    st.write(f"{mode(pred)}")