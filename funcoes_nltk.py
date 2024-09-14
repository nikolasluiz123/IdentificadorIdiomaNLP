import nltk
from nltk.lm import MLE, Laplace

from nltk.lm.preprocessing import padded_everygram_pipeline
from pandas import Series

from funcoes_treino import ResultadoTrainTestSplit, get_resultado_train_test_split


class ResultadoDadosTratados:
    def __init__(self, bigramas, vocabulario):
        self.bigramas = bigramas
        self.vocabulario = vocabulario


def identificar_idioma_laplace(dataframe_stackoverflow_portugues,
                               dataframe_stackoverflow_ingles,
                               dataframe_stackoverflow_espanhol,
                               resultado_train_test_split_testado):
    resultado_train_test_split_portugues = get_resultado_train_test_split(dataframe_stackoverflow_portugues['Questão'],
                                                                          test_size=0.2)

    resultado_train_test_split_ingles = get_resultado_train_test_split(dataframe_stackoverflow_ingles['Questão'],
                                                                       test_size=0.2)

    resultado_train_test_split_espanhol = get_resultado_train_test_split(dataframe_stackoverflow_espanhol['Questão'],
                                                                         test_size=0.2)

    modelo_portugues = treinar_modelo_laplace(resultado_train_test_split_portugues)
    modelo_ingles = treinar_modelo_laplace(resultado_train_test_split_ingles)
    modelo_espanhol = treinar_modelo_laplace(resultado_train_test_split_espanhol)

    perplexidade_portugues = calcular_perplexidade_laplace(modelo_portugues, resultado_train_test_split_testado)
    perplexidade_ingles = calcular_perplexidade_laplace(modelo_ingles, resultado_train_test_split_testado)
    perplexidade_espanhol = calcular_perplexidade_laplace(modelo_espanhol, resultado_train_test_split_testado)

    menor_perplexidade = min(perplexidade_portugues, perplexidade_ingles, perplexidade_espanhol)

    if menor_perplexidade == perplexidade_ingles:
        return "Inglês"
    elif menor_perplexidade == perplexidade_portugues:
        return "Português"
    else:
        return "Espanhol"


def calcular_perplexidade_laplace(modelo: Laplace, resultado_train_test_split: ResultadoTrainTestSplit):
    perpexidade = 0
    dados_teste_tratados = get_dados_tratados(resultado_train_test_split.dados_teste)

    for palavra in dados_teste_tratados.bigramas:
        perpexidade += modelo.perplexity(palavra)

    return perpexidade


def treinar_modelo_laplace(resultado_train_test_split: ResultadoTrainTestSplit) -> Laplace:
    dados_treino_tratados = get_dados_tratados(resultado_train_test_split.dados_treino)

    modelo = Laplace(2)
    modelo.fit(text=dados_treino_tratados.bigramas, vocabulary_text=dados_treino_tratados.vocabulario)

    return modelo


def calcular_perplexidade_mle(modelo: MLE, resultado_train_test_split: ResultadoTrainTestSplit):
    perpexidade = 0
    dados_teste_tratados = get_dados_tratados(resultado_train_test_split.dados_teste)

    for palavra in dados_teste_tratados.bigramas:
        perpexidade += modelo.perplexity(palavra)

    return perpexidade


def treinar_modelo_mle(resultado_train_test_split: ResultadoTrainTestSplit) -> MLE:
    dados_treino_tratados = get_dados_tratados(resultado_train_test_split.dados_treino)

    modelo = MLE(2)
    modelo.fit(text=dados_treino_tratados.bigramas, vocabulary_text=dados_treino_tratados.vocabulario)

    return modelo


def get_dados_tratados(questoes: Series) -> ResultadoDadosTratados:
    todas_palavras = get_palavras_questoes(questoes)
    bigramas, vocabulario = padded_everygram_pipeline(2, todas_palavras)

    return ResultadoDadosTratados(bigramas, vocabulario)


def get_palavras_questoes(coluna_dataframe) -> list[str]:
    return [palavra for questao in coluna_dataframe for palavra in nltk.word_tokenize(questao)]
