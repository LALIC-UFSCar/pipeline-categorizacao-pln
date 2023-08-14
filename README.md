# Pipeline para identificação de erros lexicais e geração de sugestões de correção
Tendo como foco a operacionalização do processamento de textos de um domínio específico, contendo possíveis termos especializados e erros de digitação/grafia produzidos
por agentes humanos ou conversão automática de formato (por exemplo, de PDF para
TXT), este artigo apresentou um pipeline de preparação/correção de textos. Esse pipeline
tem a grande vantagem de evidenciar os erros dos textos, separando o que é de fato erro e
o que não é, além de oferecer sugestões para a correção humana.

## Pré-requisitos
Antes de executar o projeto, certifique-se de ter os seguintes requisitos atendidos:

- Python 3 instalado

### O código usa algumas ferramentas auxiliares

- [Léxico do POeTISA (POrtuguese processing: Towards Syntactic Analysis and parsing)](https://sites.google.com/icmc.usp.br/poetisa)
- [JSON com lista de Países em pt-BR e seus respectivos Gentílicos, Siglas e Nome Internacional - Possui todos os países do Google Maps](https://gist.github.com/jonasruth/61bde1fcf0893bd35eea)
- [Dicionário de frequências de palavras em pt-BR disponibilizado pela Linguateca](https://www.linguateca.pt/acesso/tokens/formas.total.txt)
- [Base de termos específicos da agropecuária disponibilizado pela Embrapa](https://sistemas.sede.embrapa.br/agrotermos/)

Vale ressaltar que o arquivo `word_frequency_linguateca.txt` poossui o mesmo conteúdo que `formas.totalbr.txt` (que é o dicionário extraído da Linguateca), porém em outro formato (sem as frequências numéricas) para correta utilização por parte de uma das funções do pipeline.


## Instalação
1. Clone este repositório para o seu ambiente local:

``` shell
git clone https://github.com/LALIC-UFSCar/pie-embrapa-pln.git
```

2. Crie um ambiente virtual (opcional)

``` shell
python3 -m venv pipeline
source pipeline/bin/activate
```

3. Instale as dependências do projeto:

``` shell
pip install -r requirements.txt
```

## Uso
Para utilizar a ferramenta do pipeline é preciso somente rodar o seguinte comando indicando a pasta com os textos a serem analisados:
1. Certifique-se de ter uma pasta com os textos que deseja-se analisar.
2. Execute o seguinte comando:

``` shell
python3 pipeline_categorizacao.py <pasta-de-arquivos-txt-corpus>
```

Opcionalmente, podem ser passados mais dois parâmetros:

``` shell
python3 pipeline_categorizacao.py <pasta-de-arquivos-txt-corpus> <arquivo-unico-opcional> <anotado>
```

onde `<arquivo-unico-opcional>` indica o nome de um único arquivo de texto `txt` (caso deseje-se rodar o pipeline em apenas 1 texto) e `<anotado>` é qualquer string, indicando se o córpus possui anotação ou não. Caso o córpus seja anotado, palavras anotadas entre `<` e `>` serão desconsideradas.
