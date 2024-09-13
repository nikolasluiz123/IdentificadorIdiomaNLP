import re

import pandas as pd
from tabulate import tabulate


def get_dataframe_questoes_stackoverflow(idioma, encoding='utf-8', delimiter=','):
    dataframe = pd.read_csv(f'dataset/stackoverflow_{idioma}.csv',
                            encoding=encoding,
                            on_bad_lines='skip',
                            delimiter=delimiter)
    return dataframe


def exibir_dataframe(dataframe):
    tabela = tabulate(dataframe, headers='keys', tablefmt='grid', stralign='left', showindex=False)
    print(tabela)


def normalizar_questoes(dataframe):
    dataframe['Questão'] = dataframe['Questão'].apply(remover_tags_html_codigo)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_tags_html)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_nao_alfabeticos)
    dataframe['Questão'] = dataframe['Questão'].apply(remover_quebra_linha)
    dataframe['Questão'] = dataframe['Questão'].str.lower()


def remover_quebra_linha(texto):
    regex = re.compile(r'(\n)')
    return regex.sub('', texto)


def remover_nao_alfabeticos(texto):
    regex = re.compile(r'[^\w\s]|\d+]')
    return regex.sub('', texto)


def remover_tags_html_codigo(texto):
    regex = re.compile(r'<code>(.|\n)*?</code>')
    return regex.sub('', texto)


def remover_tags_html(texto):
    regex = re.compile(r'<.*?>')
    return regex.sub('', texto)
