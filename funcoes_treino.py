import numpy as np
from pandas import DataFrame, Series

from sklearn.model_selection import train_test_split

SEED = 1

np.random.seed(SEED)


class ResultadoTrainTestSplit:

    def __init__(self, dados_treino, dados_teste):
        self.dados_treino = dados_treino
        self.dados_teste = dados_teste


def get_resultado_train_test_split(questoes: Series, test_size: float) -> ResultadoTrainTestSplit:
    dados_treino, dados_teste = train_test_split(questoes, test_size=test_size)

    return ResultadoTrainTestSplit(dados_treino=dados_treino, dados_teste=dados_teste)
