import json

class Canonica:
    lexico = ""
    #==========================================================================
    def __init__(self, caminho_lexico):
        self.lexico = caminho_lexico
        self.leJsonLexico()
    #fim def
    #==========================================================================
    def leJsonLexico(self):
        arquivo_obj = open(self.lexico,encoding='utf-8')
        str_json = arquivo_obj.read()
        arquivo_obj.close()
        self.est_json = json.loads(str_json)
    #==========================================================================
    def consultaCanonica(self,p):
        if (self.existePalavra(p)):
            return(self.est_json[p])
        else:
            return False
        #fim if
    #fim def
    #==========================================================================
    def existePalavra(self,p):
        if (not(self.est_json.get(p))):
            return False
        else:
            return True
    #fim def
#fim class