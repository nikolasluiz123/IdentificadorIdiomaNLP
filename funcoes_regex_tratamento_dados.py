import re


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
