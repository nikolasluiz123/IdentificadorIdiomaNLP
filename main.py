import re
from funcoes import *

print('Dados do StackOverflow em PT-BT')
dataframe_stackoverflow = get_dataframe_questoes_stackoverflow(idioma='portugues')
# exibir_dataframe(dataframe_stackoverflow['Questão'].to_frame()[5:10])

print('Dados com as tags removidas')
normalizar_questoes(dataframe_stackoverflow)
exibir_dataframe(dataframe_stackoverflow['Questão'].to_frame()[10:15])

