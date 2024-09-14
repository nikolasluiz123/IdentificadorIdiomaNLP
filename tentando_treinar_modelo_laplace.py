from funcoes_nltk import treinar_modelo_laplace, \
    calcular_perplexidade_laplace
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

resultado_train_test_split_portugues = get_resultado_train_test_split(dataframe_stackoverflow_portugues['Questão'],
                                                                      test_size=0.2)

resultado_train_test_split_ingles = get_resultado_train_test_split(dataframe_stackoverflow_ingles['Questão'],
                                                                   test_size=0.2)

resultado_train_test_split_espanhol = get_resultado_train_test_split(dataframe_stackoverflow_espanhol['Questão'],
                                                                     test_size=0.2)

modelo_portugues = treinar_modelo_laplace(resultado_train_test_split_portugues)
modelo_ingles = treinar_modelo_laplace(resultado_train_test_split_ingles)
modelo_espanhol = treinar_modelo_laplace(resultado_train_test_split_espanhol)

perplexidade_portugues = calcular_perplexidade_laplace(modelo_portugues, resultado_train_test_split_portugues)
perplexidade_ingles = calcular_perplexidade_laplace(modelo_ingles, resultado_train_test_split_ingles)
perplexidade_espanhol = calcular_perplexidade_laplace(modelo_espanhol, resultado_train_test_split_espanhol)

print(f'Perplexidade Português: {perplexidade_portugues}')
print(f'Perplexidade Inglês: {perplexidade_ingles}')
print(f'Perplexidade Espanhol: {perplexidade_espanhol}')

