from sqlalchemy import create_engine
import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
from  matplotlib import pyplot as plt

us = st.secrets["us"]
pw = st.secrets["pw"]
engine = create_engine("postgresql+psycopg2://"+us+":"+pw+"@ep-dry-flower-a5zlv0k2.us-east-2.aws.neon.tech/neondb")

@st.cache_data
def prepare_and_open_data(engine=engine):
    
    engine.connect()

    clients = pd.read_sql_table('d_clients', engine)
    #work = pd.read_sql_table('d_work', engine)
    #pens = pd.read_sql_table('d_pens', engine)
    agreement = pd.read_sql_table('d_target', engine)
    job = pd.read_sql_table('d_job', engine)
    salary = pd.read_sql_table('d_salary', engine)
    last_credit = pd.read_sql_table('d_last_credit', engine)
    loan = pd.read_sql_table('d_loan', engine)
    close_loan = pd.read_sql_table('d_close_loan', engine)
    
    for tab in (agreement, job, salary, last_credit):
        tab.drop_duplicates(subset='ID_CLIENT', inplace=True)
    
    job.loc[job['WORK_TIME'] > 500, 'WORK_TIME'] = np.nan
    job['WORK_TIME'].fillna(value=job['WORK_TIME'].mean(), inplace=True)

    loan_merged = pd.merge(loan, close_loan, on='ID_LOAN', how='inner')
    loans = loan_merged.groupby('ID_CLIENT').agg({'ID_LOAN':'count', 'CLOSED_FL':'sum'}).reset_index()
    loans.rename(columns={'ID_LOAN':'LOAN_NUM_TOTAL', 'CLOSED_FL':'LOAN_NUM_CLOSED'}, inplace=True)
    df = pd.merge(clients, salary, left_on='ID', right_on='ID_CLIENT', how='left').drop('ID', axis=1)
    df = pd.merge(df, job, on='ID_CLIENT', how='left')
    df = pd.merge(df, last_credit, on='ID_CLIENT', how='left')
    df = pd.merge(df, loans, on='ID_CLIENT', how='left')
    df = pd.merge(agreement, df, on='ID_CLIENT', how='inner')

    return df 


@st.cache_data
def open_prepared_data(engine=engine):

    engine.connect()
    
    total_df = pd.read_sql_table('total_df', engine)
    
    return total_df

def target_countplot(df):
    ax = sns.countplot(df, x='TARGET', stat='percent')
    ax.bar_label(ax.containers[0], fontsize=10)
    st.pyplot(plt)

def num_features_hist_and_box(df, choice):
    for col in choice:
        fig = plt.figure(figsize = (12, 8))
        plt.subplot(2,1,1)
        sns.histplot(df[col])
        plt.grid()

        plt.subplot(2,1,2)
        sns.boxplot(df, x=col, y='TARGET', orient='h')
        plt.grid()
    
        st.table(df[col].describe())
        st.pyplot(fig)

def cat_features_countplot(df, choice):
    for col in choice:
        fig2 = plt.figure(figsize=(12,8))
        sns.countplot(df, x=col, stat='percent', hue='TARGET', order=df[col].value_counts().index)
        plt.tick_params(axis='x', labelrotation = 90)
        plt.grid()

        st.table(df[col].describe())
        st.pyplot(fig2)

def corr_matrix(df):
    fig3 = plt.figure(figsize=(12,12))
    sns.heatmap(df[df.select_dtypes(include=np.number).columns].corr(), annot=True, fmt=".1f", cmap='Blues')
    st.pyplot(fig3)

def kramer_corr_matrix(df):
    ct = pd.crosstab(df.select_dtypes(include='object').columns, df.select_dtypes(include='object').columns)
    ct.loc['TARGET', 'TARGET'] = np.nan
    for ind in ct.index:
        for col in ct.columns:
            cross_tab = pd.crosstab(df[ind], df[col])
            X2 = chi2_contingency(cross_tab, correction=False)[0]
            phi2 = X2 / cross_tab.sum().sum()
            n_rows, n_cols = cross_tab.shape
            kcor = (phi2 / min(n_cols - 1, n_rows - 1)) ** 0.5
            ct.loc[ind, col] = round(kcor, 2)
    
    fig4 = plt.figure(figsize=(7,7))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap='Blues')
    st.pyplot(fig4)

