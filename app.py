import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib import pyplot as plt
from eda import prepare_and_open_data, open_prepared_data, target_countplot, \
    num_features_hist_and_box, cat_features_countplot, corr_matrix, kramer_corr_matrix



def load_data():
    load_option = st.sidebar.radio('Загрузка данных', 
                ('Загрузить предобработанный датасет',
                 'Провести предобработку(в случае изменения исходных данных)'))
    if load_option == 'Загрузить предобработанный датасет':
        total_df = open_prepared_data()
    else:
        total_df = prepare_and_open_data()
    
    st.dataframe(total_df)
    return total_df

def main():

    st.title('Разведочный анализ данных')
    st.subheader('Предобработанный датафрейм')

    total_df = load_data()

    st.subheader('Распределение классов целевой переменной: \n (отклик клиентов на маркетинговую компанию)')
    target_countplot(total_df)
    st.write('Сильный дисбаланс классов')
    st.divider()

    st.subheader('Графики распределения числовых признаков')
    choice_num = st.pills('Числовые признаки (выберите один или несколько признаков)', 
                  (set(total_df.select_dtypes(include=np.number).columns) - {'AGREEMENT_RK', 'ID_CLIENT', 'TARGET'}), 
                  selection_mode="multi")
    num_features_hist_and_box(total_df, choice_num)  
    st.divider()

    st.subheader('Графики распределения категориальных признаков')
    choice_cat = st.pills('Категориальные признаки (выберите один или несколько признаков)', 
                  (set(total_df.select_dtypes(include='object').columns)), 
                  selection_mode="multi")
    cat_features_countplot(total_df, choice_cat)
    st.divider()

    st.subheader('Матрица корреляций')
    corr_matrix(total_df.drop(['AGREEMENT_RK', 'ID_CLIENT'], axis=1))
    st.write('Существенной корреляции целевого признака с числовыми признаками не наблюдается')
    st.write('Сильно коррелируют кол-во кредитов и закрытых кредитов; \n' 
                'имеется корреляция между характеристиками последнего кредита (сумма, срок, первый платеж); ' \
                'а также между возрастом клиента и его статусами (работает/не работает, пенсионер/не пенсионер)')
    st.divider()

    st.subheader('Матрица корреляций Крамера для категориальных переменных')
    kramer_corr_matrix(total_df)
    st.write('Существенной корреляции целевого признака с категориальными признаками также не наблюдается')
    st.write('Есть определенное взаимовлияние между характеристиками работы (отрасль, должность, направление деятельности),' \
             'а также между семейным доходом и областью проживания')
    st.divider()



if __name__ == '__main__':
        main()


