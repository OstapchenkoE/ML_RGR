import streamlit as st

st.title("Информация о наборе данных:")
st.header("Тематика датасета: Мошеничество")
st.header("Описание признаков:")
st.write("- distance_from_home - расстояние от дома. Значения: 0,004 - 10632,7")
st.write("- distance from last transaction - расстояние от последней транзакции.")
st.write("Значения: 0.000118 - 11851.1")
st.write("- ratio_to_median_purchase_price - отношение к средней цене покупки.")
st.write("Значения: 0.004399 - 267.802942")
st.write("- repeat_retailer -  повторная покупка. Значения: 0 / 1")
st.write("- used_chip -  использован чип. Значения: 0 / 1")
st.write("- used_pin_number -	использован пин код. Значения: 0 / 1")
st.write("- online_order - онлайн-заказ. Значения: 0 / 1")
st.header("Целевой параметр:")
st.write("- fraud - мошенничество. Значения: 0 / 1")

st.header(" Предобработка данных:")
st.write("Нужно предсказать будет ли совершенно мошшеничество.")
st.write("При обработке датасета была проведена нормализация данных с помощью StandardScaler.\
          Так же был исправлен дисбаланс бинарного целевого признака с помощью RandomUnderSampler.")
st.write("Были удалены выбросы.")