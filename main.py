import pandas as pd
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.figure_factory as px
import plotly.figure_factory as line
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import csv
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Estatísticas dos dAdos de desemprego em Portugal ")

st.write(
    """Estudo Efetuado pelo INE - Instiituto Nacional de Estatística sobre o estado do  dados relacionados com o Desemprego em Portugal
    """
)


with st.sidebar:
    with st.sidebar:
        st.title(" Dados Relacionados com o Desemprego, How Ai can implemento normative help and decrease categorical values on unemployment in Portugal"")
        st.title("Pode Adicionar outro daTa Set em CSV")
        st.write("Apenas Necessita de Adicionar um novo CSV")
        Button = st.button("Adicionar outro CSV")  
        if Button == True:
            File = st.file_uploader("Adcione aqui dados sobre saúde", type={"csv"})
            try:
                if File is not None:
                    df = pd.read_csv(File, low_memory=False)
            except valueError:
                print("Não Foi Adicionado CSV")

def filter_data(df: pd.DataFrame) ->pd.DataFrame:
    options = st.multiselect("escolha a Cena ", options=df.columns)
    st.write('Voçê selecionou as seguintes opções', options)
    #adicionei aqui uma cena nova
    df = pd.read_csv('Taxa de Desemprego.csv')
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.multiselect(
    'Fatores de Risco👇',
    ['Doença', 'Saude Mental', 'Estrangeiro', 'Baixa Escolaridade',
    'Carencia Economica', 'Situação de Sem Abrigo'])

       # "Escolha os Fatores 👇", df.columns,
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,
    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def filter_dataframe2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify2 = st.multiselect(
       'Fatores de Risco👇',
    ['Doença', 'Saude Mental', 'Estrangeiro', 'Baixa Escolaridade',
    'Carencia Economica', 'Situação de Sem Abrigo'])
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,
    

    if not modify2:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Selecione os Riscos", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df
    
    

#End
#df = pd.read_csv(
    #"MentalHealth.csv"
#)
#st.dataframe(filter_dataframe(df))
st.write("____________________________________________________________") 


df = pd.read_csv(
    "Taxa de Desemprego.csv"
)
#######inicio dAS TABS
tab1, tab2, tab3, tab4 , tab5 = st.tabs(["The DataFrame","The Maximum Values", "The Minumum Values", "The Average Values", "Standard Deviation"])
with tab1:
    
    st.title("Data Science for Unemployment") 
    
with tab2:
    st.header("The Maximum Values")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(" O resultado dos  dos Valores Máximos", df.max())
    with col2:
        
        df = pd.read_csv('Taxa de Desemprego.csv')
        Indx =  df.get('Date1')
        arr1  = df.get('Homens')
        arr2  = df.get('Mulheres')
        arr3  = df.get('Desempregados')
        arr4 =df.get('Ensino superior')
    
        marks_list = df['Date1'].tolist()
    
        marks_list2 = df['Desempregados'].tolist()
    
        marks_list5 = df['Homens'].tolist()
        marks_list3 = df['Mulheres'].tolist()
    
    
        marks_list4 = df['Ensino superior'].tolist()
    
        dict = {'Desempregados': marks_list2, 'Mulheres': marks_list3, 'Ensino superior': marks_list4, 'Homens' : marks_list5} 
        
        df1 = pd.DataFrame(dict)
        st.write(max(df))
        chart_data = pd.DataFrame(df, columns=["Desempregados", "Mulheres", "Ensino superior", "Homens"])
    
        st.line_chart(chart_data)
with tab3:
    st.header("The Minumum Values")
    st.write(" O resultado dos  dos Valores minimos", df.min())
    
with tab4:
    st.header("The Average Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write(" O resultado da média dos Valores é", df.mean())
    with col2:
        st.area_chart(data = df.mean())
       
with tab5:
    st.header("Standard Deviation")
    col1, col2 = st.columns(2)
    with col1:
        st.write(" O resultado da variancia", np.std(df))
    with col2:
        st.area_chart(data = np.std(df))

######FIM DAS TABS


st.write("Trabalho de Pesquisa e Programação: Paulo Ricardo Monteiro")
st.write("Formação em Fundamentos de Python Avançado por José Luis Boura - 2023/2024")
#st.line_chart(df, x=df.index, y=["Homens", "Mulheres"])

# Plot!
#st.plotly_chart(fig, use_container_width=True)

#import streamlit as st
#import streamlit.components.v1 as components
#p = open("lda.html")
#components.html(p.read(), width=1000, height=800, )
