import pandas as pd
from tabulate import tabulate

from funcoes_regex_tratamento_dados import *


def get_dataframe_questoes_stackoverflow(idioma, encoding='utf-8', delimiter=','):
    dataframe = pd.read_csv(f'dataset/stackoverflow_{idioma}.csv',
                            encoding=encoding,
                            on_bad_lines='skip',
                            delimiter=delimiter)
    return dataframe


def adicionar_coluna_idioma(df_portugues, df_ingles, df_espanhol):
    df_portugues['idioma'] = 'portugues'
    df_ingles['idioma'] = 'ingles'
    df_espanhol['idioma'] = 'espanhol'


def exibir_dataframe(dataframe):
    tabela = tabulate(dataframe, headers='keys', tablefmt='grid', stralign='left', showindex=False)
    print(tabela)


def normalizar_questoes(dataframe):
    dataframe['Questão'] = dataframe['Questão'].apply(remover_tags_html_codigo)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_tags_html)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_nao_alfabeticos)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_quebra_linha)
    dataframe['Questão'] = dataframe['Questão'].str.lower()
