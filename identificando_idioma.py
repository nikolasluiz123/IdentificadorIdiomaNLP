from funcoes_nltk import identificar_idioma_laplace
from funcoes_pandas import get_dataframe_questoes_stackoverflow, normalizar_questoes, adicionar_coluna_idioma
from funcoes_treino import get_resultado_train_test_split

dataframe_stackoverflow_portugues = get_dataframe_questoes_stackoverflow(idioma='portugues')

dataframe_stackoverflow_ingles = get_dataframe_questoes_stackoverflow(idioma='ingles')

dataframe_stackoverflow_espanhol = get_dataframe_questoes_stackoverflow(idioma='espanhol',
                                                                        encoding='cp1252',
                                                                        delimiter=';')

normalizar_questoes(dataframe_stackoverflow_portugues)
normalizar_questoes(dataframe_stackoverflow_ingles)
normalizar_questoes(dataframe_stackoverflow_espanhol)

adicionar_coluna_idioma(df_portugues=dataframe_stackoverflow_portugues,
                        df_ingles=dataframe_stackoverflow_ingles,
                        df_espanhol=dataframe_stackoverflow_espanhol)

resultado_train_test_split_testado = get_resultado_train_test_split(dataframe_stackoverflow_espanhol['Questão'],
                                                                    test_size=0.2)

idioma_identificado = identificar_idioma_laplace(dataframe_stackoverflow_portugues,
                                                 dataframe_stackoverflow_ingles,
                                                 dataframe_stackoverflow_espanhol,
                                                 resultado_train_test_split_testado)

print(f'Idioma Identificado = {idioma_identificado}')
