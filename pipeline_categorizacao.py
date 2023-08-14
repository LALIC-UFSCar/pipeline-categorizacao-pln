print("Importando bibliotecas...\n")

from Canonica import Canonica
lexico_json = "LexicoPoetisa.json"
lexico = Canonica(lexico_json)
import os
import shutil
import sys
import json
from tqdm import tqdm
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
from symspellpy import SymSpell, Verbosity
from itertools import islice
symsp = SymSpell()
symsp.load_dictionary('formas.totalbr.txt', term_index=1, count_index=0, separator='\t', encoding='ISO-8859-1')
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("word_frequency_linguateca.txt", encoding="utf8").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

# INFER SPACES ******************************************************************************
def infer_spaces(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  #******************************************************************************
  # Find the best match for the i first characters, assuming cost has
  # been built for the i-1 first characters.
  # Returns a pair (match_cost, match_length).
  def best_match(i):
    candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
    return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
  #fim def
  #******************************************************************************
  
  # Build the cost array.
  cost = [0]
  for i in range(1,len(s)+1):
    c,k = best_match(i)
    cost.append(c)
  #fim for
  
  # Backtrack to recover the minimal-cost string.
  out = []
  i = len(s)
  while i>0:
    c,k = best_match(i)
    assert c == cost[i]
    out.append(s[i-k:i])
    i -= k
  #fim for
  
  return " ".join(reversed(out))
#fim def

# GET PARAMETERS ******************************************************************************
#print('\nDigite o nome da pasta que contém os textos: ')
#pasta = input()
qtde_parametros = len(sys.argv)

if (qtde_parametros >= 2):
  # read folder name in which the texts are in
  pasta = sys.argv[1]

  # optionally run the pipeline in only one text file
  if qtde_parametros >= 3:
    arquivo_texto = sys.argv[2]
  else:
    arquivo_texto = False

  # indicates if the texts are annotated or not
  if qtde_parametros >= 4:
    anotado = True
  else:
    anotado = False
else:
  print("Erro de sintaxe!\n")
  print("Comando: python3 pipeline_de_correcao.py <pasta-de-arquivos-txt-corpus> <arquivo-unico-opcional> <anotado>")
  print("\tExemplo: python3 pipeline_de_correcao.py /home/corpus/ projeto.txt anotado")
  sys.exit()
#fim if

if not(arquivo_texto):
  try:
    textos = os.listdir(pasta)
  except:
    print("Nome de pasta inválida")
    sys.exit()
  #fim try
else:
  textos = pasta+arquivo_texto
#fim if

# verify if corpus is annotated
if (len(sys.argv[2:]) != 0):
  print('--CORPUS ANOTADO--')
  anotado = True
else:
  print('--CORPUS SEM ANOTAÇÃO--')
  anotado = False
#fim if

# GENERATE TOKENS ******************************************************************************
sentencas  = []
for filename in tqdm(textos):
  f = open(os.path.join(pasta,filename), encoding="utf8")  
  projeto = f.read()

  # separate sentences
  frases = sent_tokenize(projeto)
  for frase in frases:
    # verify words in <...>
    if anotado:
      tokens = frase.split(" ")
      processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum() and "<" not in w]
    # tokenize sentences
    else:
      tokens = nltk.word_tokenize(frase)
      processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum()]
    #fim if  
    sentencas.append(processada)
  #fim for
#fim for

from nltk.lm.preprocessing import flatten

# concatenate all tokens from texts in a list
tokens = list(flatten(sentencas))

# filter non repeated words (types)
types = list(set(tokens))

# IDENTIFY UNKNOWN WORDS ******************************************************************************
# verify if token exists in lexic
desconhecidos = []
for word in types:
  if not lexico.existePalavra(word):
    desconhecidos.append(word)
  #fim if
#fim for

'''
print("Quant. palavras do corpus: ", len(types))
print("Quant. palavras desconhecidas: ", len(desconhecidos))
print("Porcentagem de palavras desconhecidas no corpus: ", (len(desconhecidos)/len(types))*100, "%")
'''

print("\n Preparando funções para categorização das desconhecidas...")
import pandas as pd

# GENERATE DATAFRAME ******************************************************************************
# starts dataframe to categorize unknown words
def create_clear_df():
  df = pd.DataFrame(columns=['palavra', 'sugestão', 'classe'])
  df['palavra'] = desconhecidos
  df['sugestão'] = '-'
  df['classe'] = 'desconhecida'
  return df
#fim def

# READ LEXIC ****************************************************************************** 
df = create_clear_df()

import re
import unidecode

# read lexic json to string
arquivo_obj = open(lexico_json,encoding='utf-8')
str_json_lexico = arquivo_obj.read()
arquivo_obj.close()

# capture all json keys
palavras_lexico = re.findall('(?<=")(\S*)(?=":)', str_json_lexico)

# ACCENTUATION ERRORS ******************************************************************************
# unaccentuate lexic words
def desacentuando_palavra(palavra):
  return unidecode.unidecode(palavra)
#fim def

# decode words from unicode
def tirando_palavra_do_unicode(palavra_unicode):
  return palavra_unicode.encode().decode('unicode-escape')
#fim def

# unaccentuate lexic words that are in unicode
def desacentuando_palavra_unicode(palavra_unicode):
  palavra_sem_unicode = tirando_palavra_do_unicode(palavra_unicode)
  return desacentuando_palavra(palavra_sem_unicode)
#fim def

dic_palavras_lexico_sem_acento = {}

# creates a new dictionary with unaccentuated words from lexic
for palavra_acentuada in palavras_lexico:
  palavra_sem_acento = desacentuando_palavra_unicode(palavra_acentuada)
  dic_palavras_lexico_sem_acento[palavra_sem_acento] = tirando_palavra_do_unicode(palavra_acentuada)
#fim for

#PLACES FROM IBGE AND GOOGLE MAPS ******************************************************************************
from ibge.localidades import *

dados_regioes = Regioes().getNome()
dados_estados = Estados().getNome()
dados_municipios = Municipios().getNome()
#concatenate data from regions, states and cities from Brazil
localidades_ibge = dados_regioes + dados_municipios + dados_estados

# words with places from Brazil
# in lower case and without accent
localidades_ibge_lower_sem_acento = []
for localidade in localidades_ibge:
  localidade_sem_acento = desacentuando_palavra(localidade)
  localidades_ibge_lower_sem_acento.append(localidade_sem_acento.lower())
#fim for

json_paises = open("paises-gentilicos-google-maps.json", encoding="utf8")
paises = json.load(json_paises)
json_paises.close()

# list with country names and gentiles
# in lower case and without accent
nome_paises_gentilicos_lower_sem_acento = []
for pais in paises:
  pais_sem_acento = desacentuando_palavra(pais['nome_pais'])
  gentilico_sem_acento = desacentuando_palavra(pais['gentilico'])
  nome_paises_gentilicos_lower_sem_acento.append(pais_sem_acento.lower())
  nome_paises_gentilicos_lower_sem_acento.append(gentilico_sem_acento.lower())
#fim for

from googletrans import Translator
translator = Translator()

df = create_clear_df()

# SIMILAR SPELL ******************************************************************************
# function to find if word has another word with similar spell
def verifica_symspell(palavra):
  sugestao = symsp.lookup(palavra, Verbosity.CLOSEST, max_edit_distance=1, transfer_casing=True, include_unknown=True)[0].term
  if (sugestao != palavra) and lexico.existePalavra(sugestao):
    return sugestao
  else:
    return -1
  #fim if
#fim def

# AGGLUTINATED WORD ******************************************************************************
# function to find if word is a concatenation of many words
# it uses a function to infer spaces
def palavra_aglutinada_existe(palavra):
  sugestao = infer_spaces(palavra)
  lista = (sugestao).split(' ')
  ok = False
  for termo in lista:
    if (len(termo)>1):
      ok = True
      if not lexico.existePalavra(termo):
        ok = False
  #fim for
  
  if ok:
    return [sugestao]
  else:
    return ''
  #fim if
#fim def 

# TRANSLATION ******************************************************************************
# translate word to english if there is a translation
def traduz_ingles(palavra):
  try:
    translation_en = translator.translate(palavra, src='en', dest='pt')
  except:
    print("Erro traducao Ingles:",palavra)
  else:
    return translation_en.text
#fim def

# translate word to spanish if there is a translation
def traduz_espanhol(palavra):
  translation_es = translator.translate(palavra, src='es', dest='pt')
  return translation_es.text
#fim def

# translate word to french if there is a translation
def traduz_frances(palavra):
  translation_fr = translator.translate(palavra, src='fr', dest='pt')
  return translation_fr.text
#fim def

# PIPELINE FUNCTION ******************************************************************************
def categorizacao(palavra):
  palavra_sem_acento = desacentuando_palavra(palavra)
  if palavra.isnumeric():
    return ['numerico', False]
  elif (palavra_sem_acento in localidades_ibge_lower_sem_acento) or (palavra_sem_acento in nome_paises_gentilicos_lower_sem_acento):
    return ['local', False]
  elif palavra in dic_palavras_lexico_sem_acento:
    #troca pela palavra acetuada
    return ['falta_acento', dic_palavras_lexico_sem_acento[palavra]] #vai trocar pela palavra acetuada
  elif palavra_sem_acento in dic_palavras_lexico_sem_acento:
    #troca pela palavra acetuada de forma correta
    return ['acent_err', dic_palavras_lexico_sem_acento[palavra_sem_acento]]
  elif verifica_symspell(palavra) != -1:
    #troca pela palavra corrigida
    return ['symspell', verifica_symspell(palavra)]
  elif (len(palavra_aglutinada_existe(palavra)) != 0):
      palavras_desaglutinadas = palavra_aglutinada_existe(palavra)
      palavras_desaglutinadas.insert(0, 'aglutinada')
      return palavras_desaglutinadas    
  elif traduz_ingles(palavra) != palavra:
    return ['traduzida', False]
  elif traduz_espanhol(palavra) != palavra:
    return ['traduzida', False]
  elif traduz_frances(palavra) != palavra:
    return ['traduzida', False]
  else:
    return ['desconhecida', False]
  #fim if
#fim def

# EXECUTE PIPELINE ******************************************************************************
print("Começando categorização")
indice = 0

for palavra in tqdm(desconhecidos):
  palavra_categorizada = categorizacao(palavra)
  df.at[indice, 'classe'] = palavra_categorizada[0]
  
  if (palavra_categorizada[1]):
    df.at[indice, 'sugestão'] = palavra_categorizada[1]
  #fim if
  indice = indice + 1
#fim for

folder = "desconhecidas"

if not os.path.exists(folder):
  os.makedirs(folder)
else:
  shutil.rmtree(folder)
  os.makedirs(folder)
#fim if

print("Exportando o csv de desconhecidas para a pasta desconhecidas...")
pipeline_order = ['numerico', 'local', 'falta_acento', 'acent_err', 'symspell', 'aglutinada', 'traduzida', 'desconhecida']
df['classe'] = pd.Categorical(df['classe'], pipeline_order)
df.sort_values(['classe', 'palavra'], inplace=True, ignore_index=True)
df.to_csv(folder+'/desconhecidas.csv', index=False)
print("\n--FIM DO PIPELINE--")

# REWRITE TEXTS WITH PIPELINE SUGGESTIONS ******************************************************************************
print("\n--COMEÇANDO A REMONTAR OS TEXTOS--")
path_corrigidos = "textos-corrigidos"
print("Os textos remontados ficarão na pasta", path_corrigidos)

if not os.path.exists(path_corrigidos):
  os.makedirs(path_corrigidos)
else:
  shutil.rmtree(path_corrigidos)
  os.makedirs(path_corrigidos)
#fim if

subsdf = df[(df.sugestão != '-')]
subsdict = subsdf.set_index('palavra').T.to_dict('list')
error_occurr_dict = {}
mapping = {'falta_acento': 0, 'acent_err': 1, 'symspell':2, 'aglutinada':3}

for filename in tqdm(textos):
  name = filename.split('-')[0]
  error_occurr_dict[name] = [0, 0, 0, 0]
  f = open(os.path.join(pasta,filename), encoding="utf8")    
  projeto = f.read()
  
  if anotado:
    proj_corrig = projeto.replace("<", "").replace(">", "")
  else:
    proj_corrig = projeto
  #fim if
  
  for palavra in projeto.split(" "):
    if (palavra not in stopwords) and (palavra in subsdict):
      sugestao, classe = subsdict[palavra]
      proj_corrig = proj_corrig.replace(' '+palavra+' ', ' '+palavra+'['+sugestao+']'+'{'+classe+'}'+' ')
      idx = mapping[classe]
      error_occurr_dict[name][idx] += 1
    #fim if
  #fim for
  
  f = open(os.path.join(path_corrigidos,filename[:-4]+'-corrigido.txt',), 'w', encoding="utf8")
  f.write(proj_corrig)
  f.close()
#fim for

error_occurr_df = pd.DataFrame.from_dict(error_occurr_dict, orient='index')
inv_map = {v: k for k, v in mapping.items()}
error_occurr_df.rename(columns=inv_map, inplace=True)
print("Exportando err_occur para a pasta desconhecidas:")
error_occurr_df.to_csv(os.path.join(folder,'err_occurr.csv'), index_label='ID do Projeto')
print('\n--FIM--')
