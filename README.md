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
- [Dicionário de frequências de palavras em pt-BR disponibilizado pela Linguateca](https://www.linguateca.pt/COMPARA/listas_freq.php)
- Veio daqui? https://www.linguateca.pt/acesso/ordenador.php
- [Base de termos específicos da agropecuária disponibilizado pela Embrapa](https://sistemas.sede.embrapa.br/agrotermos/)
- [TO DO Miguel] Qual a diferença entre word_frequency_linguateca.txt e formas.totalbr.txt


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
python3 pipeline_categorizacao.py /pasta-com-textos
```

[TO DO Miguel] Explicar diferenças dos parâmetros

Este comando irá gerar o lexer com base no arquivo da gramática, analisar o arquivo de entrada e gerar um arquivo de saída chamado `out.txt` com os erros léxicos e semânticos encontrados.