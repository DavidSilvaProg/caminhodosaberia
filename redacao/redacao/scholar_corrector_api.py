from flask import Flask, request, jsonify
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import json

# Download NLTK resources (run once)
nltk.data.path.append("nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Critérios de correção do ENEM
CRITERIOS = {
    "competencia_1": {
        "nome": "Domínio da norma padrão da língua escrita",
        "peso": 2.0,
        "descricao": "Avalia o domínio da norma culta da língua portuguesa."
    },
    "competencia_2": {
        "nome": "Compreensão da proposta de redação",
        "peso": 2.0,
        "descricao": "Avalia se o texto atende à proposta temática e ao tipo textual solicitado."
    },
    "competencia_3": {
        "nome": "Capacidade de organizar e relacionar informações",
        "peso": 2.0,
        "descricao": "Avalia a organização lógica do texto e a articulação entre as partes."
    },
    "competencia_4": {
        "nome": "Demonstração de conhecimento da língua necessária para argumentação",
        "peso": 2.0,
        "descricao": "Avalia o uso de recursos linguísticos para construir a argumentação."
    },
    "competencia_5": {
        "nome": "Elaboração de proposta de intervenção para o problema abordado",
        "peso": 2.0,
        "descricao": "Avalia a proposta de intervenção social para o problema discutido."
    }
}

# Dicionário de temas e palavras-chave para análise
TEMAS_REDACAO = {
    "educacao": ["educação", "escola", "professor", "ensino", "aprendizado", "aluno", "universidade"],
    "meio_ambiente": ["meio ambiente", "natureza", "sustentabilidade", "poluição", "desmatamento", "recursos naturais"],
    "saude": ["saúde", "hospital", "médico", "doença", "prevenção", "SUS", "tratamento"],
    "violencia": ["violência", "segurança", "crime", "agressão", "homicídio", "polícia", "armas"],
    "tecnologia": ["tecnologia", "internet", "redes sociais", "digital", "inovação", "ciência", "robótica"]
}

# Modelos de redações nota 1000 para comparação
REDACOES_EXEMPLO = {
    "educacao": """
    A educação é o alicerce para o desenvolvimento de qualquer sociedade. No Brasil, apesar dos avanços nas últimas décadas, ainda enfrentamos desafios significativos na qualidade do ensino público. É necessário investir na formação de professores, na infraestrutura das escolas e na valorização da carreira docente para garantir um futuro melhor para as próximas gerações.

    Em primeiro lugar, a qualificação dos professores é fundamental. Muitos educadores não recebem formação continuada adequada, o que impacta diretamente na qualidade do ensino. Programas de capacitação constante, com foco em metodologias ativas de aprendizagem, poderiam melhorar esse cenário.

    Além disso, a infraestrutura das escolas públicas é precária em muitas regiões do país. Salas de aula superlotadas, falta de bibliotecas e laboratórios inadequados dificultam o processo de ensino-aprendizagem. Investimentos públicos direcionados a essas carências são urgentes.

    Por fim, a desvalorização da carreira docente leva ao desinteresse pelos cursos de licenciatura. Melhores salários e planos de carreira atrairiam jovens talentosos para a profissão, elevando a qualidade da educação.

    Diante desse cenário, é imprescindível que o governo federal, em parceria com estados e municípios, implemente um plano nacional de valorização da educação, com aumento progressivo do investimento em formação docente, infraestrutura escolar e remuneração dos professores. Somente assim construiremos uma sociedade mais justa e desenvolvida.
    """,
    "meio_ambiente": """
    A preservação do meio ambiente é um desafio global que requer ações imediatas. No Brasil, o desmatamento e as queimadas ameaçam biomas importantes como a Amazônia e o Cerrado, colocando em risco a biodiversidade e o equilíbrio climático do planeta. É urgente adotar medidas efetivas para combater esses problemas.

    Em primeiro lugar, é necessário fortalecer os órgãos de fiscalização ambiental. O Ibama e o ICMBio carecem de recursos humanos e materiais para atuar de forma eficiente. Maior orçamento e contratação de profissionais qualificados permitiriam uma atuação mais efetiva no combate aos crimes ambientais.

    Além disso, é preciso promover alternativas econômicas sustentáveis para as populações que vivem em áreas de preservação. Muitas vezes, o desmatamento ocorre por falta de opções de renda. Programas de incentivo ao ecoturismo, ao manejo florestal sustentável e à agricultura de baixo impacto poderiam mudar essa realidade.

    Outro aspecto importante é a educação ambiental. Conscientizar a população desde a escola sobre a importância da preservação é fundamental para formar cidadãos responsáveis. Campanhas públicas e inclusão do tema no currículo escolar são medidas necessárias.

    Portanto, a proteção do meio ambiente exige uma ação conjunta do poder público, setor privado e sociedade civil. O governo deve liderar esse processo, criando políticas públicas eficientes, enquanto cada cidadão deve fazer sua parte, adotando hábitos sustentáveis no dia a dia. Só assim garantiremos um futuro para as próximas gerações.
    """
}

def preprocess_text(text):
    """Pré-processamento do texto para análise"""
    # Converter para minúsculas
    text = text.lower()
    # Remover pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizar
    tokens = word_tokenize(text)
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words]
    # Juntar tokens novamente
    text = ' '.join(tokens)
    return text

def calcular_similaridade(texto1, texto2):
    """Calcula a similaridade entre dois textos usando TF-IDF e cosseno"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([texto1, texto2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def detectar_tema(texto):
    """Detecta o tema principal da redação com base nas palavras-chave"""
    texto_preprocessado = preprocess_text(texto)
    scores = {}
    
    for tema, palavras_chave in TEMAS_REDACAO.items():
        score = 0
        for palavra in palavras_chave:
            if palavra in texto_preprocessado:
                score += 1
        scores[tema] = score
    
    tema_detectado = max(scores, key=scores.get)
    return tema_detectado if scores[tema_detectado] > 0 else "outros"

def analisar_competencia_1(texto):
    """Analisa o domínio da norma padrão da língua escrita"""
    # Contar erros gramaticais simples (simulação)
    erros = 0
    
    # Verificar parágrafos (deve ter pelo menos 4)
    paragrafos = [p for p in texto.split('\n') if p.strip()]
    if len(paragrafos) < 4:
        erros += (4 - len(paragrafos)) * 2
    
    # Verificar concordância básica (simplificado)
    palavras = texto.lower().split()
    for i in range(len(palavras)-1):
        if palavras[i] == "a" and palavras[i+1] in ["meninos", "homens"]:
            erros += 1
        if palavras[i] == "o" and palavras[i+1] in ["meninas", "mulheres"]:
            erros += 1
    
    # Pontuação (verificar pontos finais)
    frases = re.split(r'[.!?]', texto)
    if len(frases) < 5:
        erros += 2
    
    # Normalizar erros para nota entre 0 e 200
    if erros == 0:
        return 200
    elif erros <= 3:
        return 160
    elif erros <= 6:
        return 120
    elif erros <= 9:
        return 80
    else:
        return 40

def analisar_competencia_2(texto, tema):
    """Analisa a compreensão da proposta de redação"""
    # Verificar se o texto trata do tema detectado
    tema_detectado = detectar_tema(texto)
    
    if tema_detectado != tema:
        return 40  # Nota baixa se fugir ao tema
    
    # Verificar se é dissertativo-argumentativo
    marcadores_argumentativos = ["em primeiro lugar", "além disso", "por outro lado", 
                               "no entanto", "portanto", "dessa forma", "assim sendo"]
    
    marcadores_presentes = sum(1 for marcador in marcadores_argumentativos 
                             if marcador in texto.lower())
    
    if marcadores_presentes >= 3:
        return 200
    elif marcadores_presentes >= 1:
        return 120
    else:
        return 80

def analisar_competencia_3(texto):
    """Analisa a capacidade de organizar e relacionar informações"""
    # Verificar estrutura lógica (introdução, desenvolvimento, conclusão)
    paragrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    
    if len(paragrafos) < 3:
        return 60
    
    # Verificar progressão temática
    similaridade_paragrafos = []
    for i in range(len(paragrafos)-1):
        sim = calcular_similaridade(paragrafos[i], paragrafos[i+1])
        similaridade_paragrafos.append(sim)
    
    avg_similaridade = np.mean(similaridade_paragrafos) if similaridade_paragrafos else 0
    
    if 0.3 <= avg_similaridade <= 0.7:
        return 200
    elif avg_similaridade < 0.3:
        return 120  # Pouca coesão
    else:
        return 80   # Muita repetição

def analisar_competencia_4(texto):
    """Analisa conhecimento linguístico para argumentação"""
    # Contar conectivos e operadores argumentativos
    conectivos = ["portanto", "assim", "logo", "pois", "porque", "embora", 
                 "entretanto", "no entanto", "todavia", "além disso"]
    
    qtd_conectivos = sum(texto.lower().count(conectivo) for conectivo in conectivos)
    
    # Verificar variedade lexical
    palavras = preprocess_text(texto).split()
    vocabulario = set(palavras)
    taxa_vocabulario = len(vocabulario) / len(palavras) if palavras else 0
    
    if qtd_conectivos >= 5 and taxa_vocabulario > 0.5:
        return 200
    elif qtd_conectivos >= 3 and taxa_vocabulario > 0.4:
        return 160
    elif qtd_conectivos >= 1 and taxa_vocabulario > 0.3:
        return 120
    else:
        return 80

def analisar_competencia_5(texto, tema):
    """Analisa a proposta de intervenção"""
    # Verificar se há proposta de intervenção no último parágrafo
    paragrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    if not paragrafos:
        return 0
    
    ultimo_paragrafo = paragrafos[-1].lower()
    
    # Palavras-chave que indicam proposta de intervenção
    palavras_chave = ["solução", "medida", "proposta", "intervenção", "ação", 
                     "sugere-se", "é necessário", "deve-se", "precisa-se"]
    
    tem_proposta = any(palavra in ultimo_paragrafo for palavra in palavras_chave)
    
    if not tem_proposta:
        return 40
    
    # Verificar se a proposta é detalhada e relacionada ao tema
    tema_detectado = detectar_tema(ultimo_paragrafo)
    if tema_detectado != tema:
        return 80
    
    # Verificar se menciona agentes (governo, sociedade, etc.)
    agentes = ["governo", "estado", "município", "sociedade", "escola", 
              "família", "empresas", "ong", "população"]
    
    qtd_agentes = sum(ultimo_paragrafo.count(agente) for agente in agentes)
    
    if qtd_agentes >= 2:
        return 200
    elif qtd_agentes >= 1:
        return 160
    else:
        return 120

def gerar_comentarios(notas, tema):
    """Gera comentários personalizados para cada competência"""
    comentarios = {}
    
    # Competência 1
    if notas["competencia_1"] >= 180:
        comentarios["competencia_1"] = "Excelente domínio da norma padrão da língua escrita, com pouquíssimos ou nenhum desvio gramatical."
    elif notas["competencia_1"] >= 120:
        comentarios["competencia_1"] = "Bom domínio da norma padrão, com alguns desvios pontuais que não comprometem a compreensão do texto."
    else:
        comentarios["competencia_1"] = "Necessita melhorar o domínio da norma padrão, com desvios gramaticais que dificultam a compreensão em alguns trechos."
    
    # Competência 2
    if notas["competencia_2"] >= 180:
        comentarios["competencia_2"] = f"Ótima compreensão da proposta de redação sobre {tema}, com texto perfeitamente adequado ao tipo dissertativo-argumentativo."
    elif notas["competencia_2"] >= 120:
        comentarios["competencia_2"] = f"Boa compreensão da proposta sobre {tema}, com texto adequado ao tipo dissertativo-argumentativo, mas com pequenos desvios."
    else:
        comentarios["competencia_2"] = f"Necessita melhorar a compreensão da proposta sobre {tema}. O texto pode ter fugido parcialmente ao tema ou ao tipo dissertativo-argumentativo."
    
    # Competência 3
    if notas["competencia_3"] >= 180:
        comentarios["competencia_3"] = "Excelente organização e relação de informações, com argumentos bem articulados e progressão temática coerente."
    elif notas["competencia_3"] >= 120:
        comentarios["competencia_3"] = "Boa organização das ideias, com argumentos articulados, mas poderia melhorar a progressão temática em alguns trechos."
    else:
        comentarios["competencia_3"] = "Necessita melhorar a organização das ideias. Os argumentos podem estar pouco articulados ou a progressão temática pode estar confusa."
    
    # Competência 4
    if notas["competencia_4"] >= 180:
        comentarios["competencia_4"] = "Excelente demonstração de conhecimento linguístico, com boa variedade de conectivos e coerência argumentativa."
    elif notas["competencia_4"] >= 120:
        comentarios["competencia_4"] = "Bom conhecimento linguístico, com uso adequado de conectivos, mas poderia diversificar mais os recursos argumentativos."
    else:
        comentarios["competencia_4"] = "Necessita melhorar o uso de recursos linguísticos para argumentação. Faltam conectivos ou a argumentação pode estar pouco desenvolvida."
    
    # Competência 5
    if notas["competencia_5"] >= 180:
        comentarios["competencia_5"] = f"Excelente proposta de intervenção para o problema de {tema}, bem detalhada e com agentes claramente definidos."
    elif notas["competencia_5"] >= 120:
        comentarios["competencia_5"] = f"Boa proposta de intervenção para {tema}, mas poderia ser mais detalhada ou mencionar mais agentes envolvidos."
    else:
        comentarios["competencia_5"] = f"Necessita melhorar a proposta de intervenção para {tema}. Pode estar genérica, pouco clara ou desvinculada do tema."
    
    return comentarios

@app.route('/corrigir-redacao', methods=['POST'])
def corrigir_redacao():
    """Endpoint para correção de redação"""
    try:
        data = request.get_json()
        
        if not data or 'texto' not in data or 'tema' not in data:
            return jsonify({'error': 'Dados incompletos. Envie "texto" e "tema".'}), 400

        # Processamento da correção
        texto = data['texto']
        tema = data['tema']

        # Análise das competências
        notas = {
            "competencia_1": analisar_competencia_1(texto),
            "competencia_2": analisar_competencia_2(texto, tema),
            "competencia_3": analisar_competencia_3(texto),
            "competencia_4": analisar_competencia_4(texto),
            "competencia_5": analisar_competencia_5(texto, tema)
        }

        # Cálculo da nota final
        nota_final = sum(notas.values())

        # Gerar comentários
        comentarios = gerar_comentarios(notas, tema)

        # Resposta da API
        return jsonify({
            'nota_final': nota_final,
            'notas': notas,
            'comentarios': comentarios,
            'criterios': CRITERIOS,
            'texto': texto,
            'tema': tema
        })

    except Exception as e:
        print(f"Erro: {e}")  # Mostra erro no terminal
        return jsonify({'error': 'Erro interno no servidor.', 'detalhes': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)