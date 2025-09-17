# app.py
from flask import Flask, request, jsonify
import os
import re
import json
import string
import traceback
import math
import unicodedata
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK (com fallbacks – não baixa nada em runtime)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)

@app.get("/healthz")
def healthz():
	return {"ok": True}

# =========================
#  CONFIG / ARQUIVOS
# =========================
REDACOES_JSON_PATH = os.environ.get("REDACOES_JSON_PATH", "redacoes_exemplo.json")
VERSAO_CORRETOR = "2.1.0"  # temas livres + suporte a título do front + C2/C5 sem catálogo fixo

def load_redacoes_exemplo():
    """
    Carrega redacoes_exemplo.json apenas para vetor auxiliar/diagnóstico.
    NÃO influencia a nota nas competências (temas livres).
    """
    try:
        with open(REDACOES_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("redacoes_exemplo.json inválido: raiz deve ser objeto (dict).")
            return data
    except FileNotFoundError:
        return {
            "educacao": [
                "Texto exemplo sobre educação com estrutura temática e proposta de intervenção..."
            ],
            "meio_ambiente": [
                "Texto exemplo sobre meio ambiente, desmatamento, fiscalização e educação ambiental..."
            ]
        }
    except Exception as e:
        print(f"[AVISO] Falha ao carregar {REDACOES_JSON_PATH}: {e}")
        return {}

REDACOES_EXEMPLO = load_redacoes_exemplo()

# =========================
#  CRITÉRIOS
# =========================
CRITERIOS = {
    "competencia_1": {"nome": "Domínio da norma padrão da língua escrita", "peso": 2.0, "descricao": "Avalia o domínio da norma culta da língua portuguesa."},
    "competencia_2": {"nome": "Compreensão da proposta de redação", "peso": 2.0, "descricao": "Avalia se o texto atende à proposta temática e ao tipo textual solicitado."},
    "competencia_3": {"nome": "Capacidade de organizar e relacionar informações", "peso": 2.0, "descricao": "Avalia a organização lógica do texto e a articulação entre as partes."},
    "competencia_4": {"nome": "Demonstração de conhecimento da língua necessária para argumentação", "peso": 2.0, "descricao": "Avalia o uso de recursos linguísticos para construir a argumentação."},
    "competencia_5": {"nome": "Elaboração de proposta de intervenção para o problema abordado", "peso": 2.0, "descricao": "Avalia a proposta de intervenção social para o problema discutido."}
}

# =========================
#  (CATÁLOGO AUXILIAR – NÃO PONTUA)
# =========================
TEMAS_REDACAO = {
    # Mantido apenas para o detector auxiliar/diagnóstico (não afeta nota).
    "educacao": ["educação", "escola", "professor", "ensino", "aprendizado", "aluno", "universidade"],
    "meio_ambiente": ["meio ambiente", "natureza", "sustentabilidade", "poluição", "desmatamento", "recursos naturais"],
    "saude": ["saúde", "hospital", "médico", "doença", "prevenção", "sus", "tratamento"],
    "violencia": ["violência", "segurança", "crime", "agressão", "homicídio", "polícia", "armas"],
    "tecnologia": ["tecnologia", "internet", "redes sociais", "digital", "inovação", "ciência", "robótica"],
    "trabalho_domestico_mulher": [
        "trabalho doméstico", "trabalho de cuidado", "invisibilidade", "mulheres",
        "igualdade de gênero", "machismo", "afazeres domésticos", "papel da mulher"
    ]
}

# =========================
#  NLTK FALLBACKS
# =========================
def have_nltk_resource(path: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False

_HAS_PUNKT = have_nltk_resource('tokenizers/punkt')
_HAS_STOPWORDS = have_nltk_resource('corpora/stopwords')

def safe_word_tokenize(text: str):
    if _HAS_PUNKT:
        try:
            return word_tokenize(text, language='portuguese')
        except Exception:
            pass
    return re.findall(r"\w+|\S", text, flags=re.UNICODE)

def safe_sent_tokenize(text: str):
    if _HAS_PUNKT:
        try:
            return sent_tokenize(text, language='portuguese')
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def get_portuguese_stopwords():
    if _HAS_STOPWORDS:
        try:
            return set(stopwords.words('portuguese'))
        except Exception:
            pass
    return {
        'a','o','e','de','da','do','das','dos','um','uma','uns','umas','em','no','na','nos','nas',
        'para','por','com','sem','sobre','entre','como','que','se','ao','à','às','aos','ou','mas'
    }

STOPWORDS_PT = get_portuguese_stopwords()

# =========================
#  UTIL – COERÇÃO / NORMALIZAÇÃO
# =========================
def coerce_to_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ('texto', 'text', 'conteudo', 'content', 'body', 'resposta', 'exemplo', 'titulo', 'title'):
            v = x.get(k)
            if isinstance(v, str):
                return v
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    if isinstance(x, (list, tuple)):
        return "\n".join(coerce_to_text(e) for e in x)
    return str(x)

def normalize_text_basic(text) -> str:
    if not isinstance(text, str):
        text = coerce_to_text(text)
    text = text.lower()
    table = str.maketrans('', '', string.punctuation.replace('-', ''))
    return text.translate(table)

def preprocess_text_for_ttr(text) -> list:
    text_norm = normalize_text_basic(text)
    tokens = [t for t in re.findall(r"\w+", text_norm, flags=re.UNICODE) if t]
    return [t for t in tokens if t not in STOPWORDS_PT]

def preprocess_text_for_tfidf(text) -> str:
    return normalize_text_basic(text)

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = coerce_to_text(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

# =========================
#  ADERÊNCIA DE TEMA (TEMAS LIVRES) — considera TÍTULO + TEXTO
# =========================
def theme_adherence_score(texto: str, tema_informado: str, titulo: str = "") -> float:
    """
    Mede a aderência entre a redação (título + corpo) e a frase do tema informado (TEMAS LIVRES).
    - Usa TF-IDF com 2 documentos: [titulo+'\n\n'+texto, tema].
    - Retorna coseno em [0,1].
    - É robusto a acentos/caixa; não depende de catálogo fixo.
    """
    # combina título e corpo dando um pequeno "peso" ao título por aparecer antes
    titulo = coerce_to_text(titulo or "").strip()
    texto_full = (titulo + "\n\n" if titulo else "") + coerce_to_text(texto or "")
    texto_proc = preprocess_text_for_tfidf(texto_full)
    tema_proc  = preprocess_text_for_tfidf(coerce_to_text(tema_informado))

    if len(tema_proc.split()) == 0:
        return 0.0
    try:
        v = TfidfVectorizer()
        X = v.fit_transform([texto_proc, tema_proc])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

# =========================
#  GATE DE QUALIDADE (ZERO)
# =========================
def is_probably_gibberish(text: str) -> bool:
    t = coerce_to_text(text)
    if not t or not t.strip():
        return True

    only_letters = re.findall(r"[a-zà-ú]", normalize_text_basic(t), flags=re.IGNORECASE)
    letters_ratio = (len(only_letters) / max(1, len(t)))
    if letters_ratio < 0.55:
        return True

    vowels = re.findall(r"[aeiouáéíóúâêôãõ]", normalize_text_basic(t))
    if (len(vowels) / max(1, len(only_letters))) < 0.25:
        return True

    if re.search(r"(.)\1{6,}", t):
        return True

    return False

def should_hard_zero(texto: str) -> bool:
    """
    Regra para zerar (TEMAS LIVRES):
    - Texto muito curto (palavras < 80) OU frases < 3 OU parágrafos < 2
    - OU detecção de ruído (gibberish)
    NÃO zera por “fuga de tema”.
    """
    t = coerce_to_text(texto)
    tokens = re.findall(r"\w+", normalize_text_basic(t))
    palavras = len(tokens)
    frases = len(safe_sent_tokenize(t))
    paragrafos = len([p for p in t.split('\n') if p.strip()])

    if palavras < 80 or frases < 3 or paragrafos < 2:
        return True

    if is_probably_gibberish(t):
        return True

    return False

# =========================
#  TF-IDF – DETECÇÃO DE TEMA (AUXILIAR, NÃO USADO PARA NOTA)
# =========================
THEME_VECTOR_DATA = None

def build_theme_vectorizer():
    """
    Mantido apenas como detector auxiliar/diagnóstico (não pontua).
    """
    global THEME_VECTOR_DATA
    temas = []
    corpus = []

    for tema, exemplos in REDACOES_EXEMPLO.items():
        if isinstance(exemplos, (str, dict)):
            exemplos = [exemplos]
        elif not isinstance(exemplos, (list, tuple)):
            exemplos = [exemplos]

        for exemplo in exemplos:
            try:
                txt = coerce_to_text(exemplo)
            except Exception as e:
                print(f"[WARN] Exemplo de tema '{tema}' não convertido para texto: {type(exemplo)} - {e}")
                continue
            if txt and txt.strip():
                temas.append(tema)
                corpus.append(preprocess_text_for_tfidf(txt))

    for tema, termos in TEMAS_REDACAO.items():
        temas.append(tema)
        corpus.append(preprocess_text_for_tfidf(" ".join(termos)))

    if not corpus:
        temas = ["outros"]
        corpus = ["texto geral"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    THEME_VECTOR_DATA = {"temas": temas, "vectorizer": vectorizer, "X": X}

def detect_theme_tfidf(texto: str) -> (str, float):
    """
    Apenas informativo para debug/telemetria: NÃO influencia C2/C5.
    """
    if THEME_VECTOR_DATA is None:
        build_theme_vectorizer()
    vec = THEME_VECTOR_DATA["vectorizer"]
    X = THEME_VECTOR_DATA["X"]
    temas = THEME_VECTOR_DATA["temas"]
    q = vec.transform([preprocess_text_for_tfidf(texto)])
    sims = cosine_similarity(q, X)[0]
    agg = {}
    for i, tema in enumerate(temas):
        agg.setdefault(tema, []).append(sims[i])
    tema_scores = {t: float(np.mean(v)) for t, v in agg.items()}
    tema_detectado = max(tema_scores, key=tema_scores.get)
    score = tema_scores[tema_detectado]
    if score < 0.12:
        return "outros", 0.0
    return tema_detectado, score

# =========================
#  MTLD (DIVERSIDADE LEXICAL)
# =========================
def mtld(tokens, ttr_threshold=0.72):
    if not tokens or len(tokens) < 50:
        return 0.0
    def _mtld_calc(seq):
        factors = 0
        types = set()
        token_count = 0
        for t in seq:
            token_count += 1
            types.add(t)
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            partial = (1 - (len(types) / token_count)) / (1 - ttr_threshold)
            factors += partial
        if factors == 0:
            return 0.0
        return len(seq) / factors
    forward = _mtld_calc(tokens)
    backward = _mtld_calc(list(reversed(tokens)))
    if forward == 0.0 or backward == 0.0:
        return max(forward, backward)
    return (forward + backward) / 2.0

# =========================
#  COMPETÊNCIAS (TEMAS LIVRES)
# =========================
def analisar_competencia_1(texto: str) -> int:
    t = coerce_to_text(texto)
    paragrafos = [p for p in t.split('\n') if p.strip()]
    frases = safe_sent_tokenize(t)
    palavras = normalize_text_basic(t).split()

    if len(frases) < 3 or len(paragrafos) < 2 or len(palavras) < 120:
        return 60

    erros = 0
    if len(paragrafos) < 4:
        erros += (4 - len(paragrafos)) * 2.0
    if len(frases) < 5:
        erros += 3

    for i in range(len(palavras)-1):
        if palavras[i] == "a" and palavras[i+1] in ["meninos", "homens"]:
            erros += 1
        if palavras[i] == "o" and palavras[i+1] in ["meninas", "mulheres"]:
            erros += 1

    erros = min(erros, 15)
    score = max(0.0, 1.0 - (erros / 15.0))
    nota = round(score * 200)

    if len(frases) >= 8 and len(paragrafos) >= 4 and len(palavras) >= 180 and erros <= 2:
        return min(200, nota)

    if len(frases) < 5 or len(paragrafos) < 3:
        nota = min(nota, 120)
    return nota

def analisar_competencia_2(texto: str, tema_informado: str, titulo: str = "") -> int:
    """
    C2 (temas livres):
    - Mede aderência (TF‑IDF) entre TEMA INFORMADO e (TÍTULO + TEXTO).
    - Marcadores argumentativos aumentam a nota.
    - Sem penalidade por catálogo fixo.
    """
    texto_l = normalize_text_basic(texto)
    marcadores = [
        "em primeiro lugar","além disso","por outro lado","no entanto",
        "portanto","dessa forma","assim sendo","logo","todavia","entretanto"
    ]
    marcadores_presentes = sum(1 for m in marcadores if m in texto_l)

    aderencia = theme_adherence_score(texto, tema_informado, titulo)  # 0..1

    # fast-path de excelência: bom alinhamento + estrutura argumentativa
    if aderencia >= 0.25 and marcadores_presentes >= 4:
        return 200

    s_marc = 0.0 if marcadores_presentes == 0 else min(1.0, marcadores_presentes / 5.0)
    # dá um pouco mais de peso para aderência (que já considera o título)
    base = 0.72 * aderencia + 0.28 * s_marc
    nota = round(base * 200)

    if marcadores_presentes == 0 and aderencia < 0.10:
        nota = min(nota, 90)

    return max(0, min(200, nota))

def analisar_competencia_3(texto: str) -> int:
    paragrafos = [p.strip() for p in coerce_to_text(texto).split('\n') if p.strip()]
    if len(paragrafos) < 3:
        return 40

    pars_proc = [preprocess_text_for_tfidf(p) for p in paragrafos]
    try:
        v = TfidfVectorizer()
        X = v.fit_transform(pars_proc)
    except ValueError:
        return 80

    sims = []
    for i in range(len(paragrafos) - 1):
        s = cosine_similarity(X[i:i+1], X[i+1:i+2])[0][0]
        sims.append(s)
    avg_sim = float(np.mean(sims)) if sims else 0.0

    repeticao_forte = False
    for i in range(len(paragrafos)):
        for j in range(i + 1, len(paragrafos)):
            s = cosine_similarity(X[i:i+1], X[j:j+1])[0][0]
            if s > 0.995:
                repeticao_forte = True
                break
        if repeticao_forte:
            break

    if not repeticao_forte and len(paragrafos) >= 4 and 0.30 <= avg_sim <= 0.80:
        return 200

    if avg_sim <= 0.10:
        base = 0.40
    elif avg_sim >= 0.95:
        base = 0.35
    else:
        base = 0.58 + 0.42 * ((avg_sim - 0.10) / (0.95 - 0.10))
        base = max(0.0, min(0.98, base))

    if len(paragrafos) >= 4 and 0.30 <= avg_sim <= 0.80 and not repeticao_forte:
        base = max(base, 0.96)

    if repeticao_forte:
        base -= 0.12

    base = max(0.0, min(1.0, base))
    return max(0, min(200, round(base * 200)))

def analisar_competencia_4(texto: str) -> int:
    conectivos = [
        "portanto","assim","logo","pois","porque","embora",
        "entretanto","no entanto","todavia","além disso","dessa forma","desse modo"
    ]
    texto_l = normalize_text_basic(texto)
    tokens_ttr = preprocess_text_for_ttr(texto)
    total_palavras = max(1, len(re.findall(r"\w+", normalize_text_basic(texto))))
    qtd_conectivos = sum(texto_l.count(c) for c in conectivos)

    conectivos_por_100 = (qtd_conectivos / total_palavras) * 100.0

    mtld_val = mtld(tokens_ttr)
    mtld_norm = max(0.0, min(1.0, mtld_val / 45.0))
    conn_norm = max(0.0, min(1.0, conectivos_por_100 / 2.5))

    score = 0.60 * mtld_norm + 0.40 * conn_norm
    nota = round(score * 200)

    if qtd_conectivos == 0:
        return min(nota, 120)

    if qtd_conectivos >= 5 and mtld_val >= 40:
        return 200

    if qtd_conectivos >= 6 and mtld_val >= 35:
        nota = max(nota, 190)

    return max(0, min(200, nota))

def analisar_competencia_5(texto: str) -> int:
    """
    C5 (temas livres):
    - Avalia SOMENTE a completude da proposta (Agente, Ação, Meio, Finalidade, Monitoramento).
    - NÃO penaliza por desalinhamento com um catálogo de temas.
    - Se não há proposta explícita → 40.
    """
    paragrafos = [p.strip() for p in coerce_to_text(texto).split('\n') if p.strip()]
    if not paragrafos:
        return 0
    ultimos = paragrafos[-2:] if len(paragrafos) >= 2 else [paragrafos[-1]]
    cand = " ".join(ultimos).lower()

    chaves = [
        "solução","medida","proposta","intervenção","ação",
        "sugere-se","é necessário","deve-se","precisa-se","implementação","política pública",
        "é fundamental que","propõe-se","recomenda-se","deve-se garantir"
    ]
    if not any(w in cand for w in chaves):
        return 40

    agentes = ["governo","estado","município","sociedade","escola","família",
               "empresas","ong","população","comunidade","ministério","secretaria","congresso"]
    verbos_acao = ["implementar","promover","garantir","fiscalizar","criar","ampliar","ofertar","regulamentar","incentivar"]
    meios = ["por meio","com ", "via ", "programa", "plano", "campanha", "parceria"]
    finalidade = ["a fim de", "para que", "de modo a", "com o objetivo de"]
    monitor = ["prazo","meta","indicadores","monitoramento","avaliação"]

    comp = {
        "agente": any(a in cand for a in agentes),
        "acao": any(v in cand for v in verbos_acao),
        "meio": any(m in cand for m in meios),
        "finalidade": any(f in cand for f in finalidade),
        "monitoramento": any(m in cand for m in monitor)
    }
    score_comp = sum(comp.values())  # 0..5

    if score_comp >= 4:
        return 200
    if score_comp == 3:
        return 180
    if score_comp == 2:
        return 140
    return 100

# =========================
#  COMENTÁRIOS / NOTA FINAL
# =========================
def gerar_comentarios(notas, tema):
    c = {}
    c["competencia_1"] = ("Excelente domínio da norma padrão, com fluidez e poucos desvios."
        if notas["competencia_1"] >= 180 else
        "Bom domínio da norma padrão; revise pequenos desvios e pontuação."
        if notas["competencia_1"] >= 120 else
        "Melhore a correção gramatical e a pontuação. Releia focando concordância e divisão de frases."
    )
    c["competencia_2"] = (f"Ótima aderência ao tema '{tema}' e ao gênero dissertativo-argumentativo."
        if notas["competencia_2"] >= 180 else
        f"Boa aderência ao tema '{tema}'. Reforce a estrutura argumentativa com mais conectivos."
        if notas["competencia_2"] >= 120 else
        f"Atenção ao tema '{tema}' e ao gênero dissertativo-argumentativo. Use marcadores lógicos para guiar a leitura."
    )
    c["competencia_3"] = ("Ótima organização; parágrafos encadeados com progressão temática clara."
        if notas["competencia_3"] >= 180 else
        "Boa organização; pode melhorar a transição entre parágrafos e evitar repetições."
        if notas["competencia_3"] >= 120 else
        "A organização do texto pode ser aprimorada. Trabalhe a progressão de ideias e a conexão entre parágrafos."
    )
    c["competencia_4"] = ("Vocabulário variado e uso eficaz de conectivos. Argumentação consistente."
        if notas["competencia_4"] >= 180 else
        "Bom uso de conectivos e vocabulário adequado. Busque maior variedade lexical."
        if notas["competencia_4"] >= 120 else
        "Amplie a diversidade lexical e use conectivos para relacionar as ideias de forma mais clara."
    )
    c["competencia_5"] = (f"Excelente proposta de intervenção relacionada a {tema}, com agentes e etapas bem definidos."
        if notas["competencia_5"] >= 180 else
        f"Boa proposta de intervenção para {tema}, mas pode detalhar melhor agentes e meios."
        if notas["competencia_5"] >= 120 else
        f"Elabore uma proposta de intervenção mais clara e detalhada para {tema}, indicando agentes e meios."
    )
    return c

def calcular_nota_final(notas: dict) -> float:
    total_peso = sum(CRITERIOS[k]["peso"] for k in CRITERIOS.keys())
    acumulado = 0.0
    for k, v in notas.items():
        peso = CRITERIOS[k]["peso"]
        acumulado += v * peso
    score_norm = (acumulado / (200.0 * total_peso)) * 1000.0
    return round(float(score_norm), 2)

def fator_severidade(texto: str, notas: dict) -> float:
    """
    Penalização leve para falhas estruturais graves.
    Não penaliza textos excelentes com estrutura robusta e todas competências >= 150.
    """
    t = coerce_to_text(texto)
    palavras = len(re.findall(r"\w+", normalize_text_basic(t)))
    paragrafos = [p for p in t.split('\n') if p.strip()]
    frases = safe_sent_tokenize(t)

    if min(notas.values()) >= 150 and palavras >= 180 and len(paragrafos) >= 4 and len(frases) >= 8:
        return 1.00

    fails = 0
    if len(paragrafos) < 3: fails += 1
    if len(frases) < 5: fails += 1
    if palavras < 150: fails += 1
    if notas.get("competencia_5", 0) <= 60: fails += 1

    if fails >= 3:
        return 0.70
    if fails == 2:
        return 0.80
    if fails == 1:
        return 0.90
    return 1.00

def boost_excelente(notas: dict) -> float:
    tops = sum(1 for v in notas.values() if v >= 180)
    if tops >= 4:
        return 1.10
    if tops == 3:
        return 1.07
    return 1.00

def curva_topo_suave(nota: float) -> float:
    x = max(0.0, min(1000.0, nota)) / 1000.0
    y = 1.0 / (1.0 + math.exp(-7.0 * (x - 0.82)))
    val = 1000.0 * (0.85 * x + 0.15 * y)
    return val

# =========================
#  HEALTH / HELP
# =========================
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"}), 200

@app.route('/corrigir-redacao', methods=['GET'])
def corrigir_redacao_get():
    return jsonify({
        "message": "Use POST com JSON: { texto: str, tema: str, titulo?: str, aderencia_cliente?: number }",
        "exemplo": {"texto": "minha redação...", "tema": "meio ambiente", "titulo": "Título opcional"}
    }), 200

# =========================
#  ENDPOINT PRINCIPAL
# =========================
@app.route('/corrigir-redacao', methods=['POST'])
def corrigir_redacao():
    try:
        data = request.get_json(silent=True)
        if not data or 'texto' not in data or 'tema' not in data:
            return jsonify({'error': 'Dados incompletos. Envie "texto" e "tema".'}), 400

        texto  = coerce_to_text(data.get('texto', ''))
        tema   = coerce_to_text(data.get('tema', ''))
        titulo = coerce_to_text(data.get('titulo', '')).strip()  # NOVO: opcional vindo do front
        aderencia_cliente = data.get('aderencia_cliente', None)  # NOVO: heurística do front (opcional)

        # ======= HARD ZERO =======
        if should_hard_zero(texto):
            diagnostico_estrutural = {
                'palavras': len(re.findall(r"\w+", normalize_text_basic(texto))),
                'paragrafos': len([p for p in texto.split('\n') if p.strip()]),
                'frases': len(safe_sent_tokenize(texto))
            }
            return jsonify({
                'versao_corretor': VERSAO_CORRETOR,
                'gerado_em': datetime.utcnow().isoformat() + 'Z',
                'nota_final': 0.0,
                'notas': {
                    'competencia_1': 0,
                    'competencia_2': 0,
                    'competencia_3': 0,
                    'competencia_4': 0,
                    'competencia_5': 0
                },
                'comentarios': {
                    'geral': 'Texto insuficiente ou aleatório. Escreva um texto dissertativo-argumentativo com introdução, desenvolvimento, conclusão e proposta de intervenção.'
                },
                'criterios': CRITERIOS,
                'tema_informado': tema,
                'tema_detectado': tema,  # temas livres: detectado = informado
                'score_tema': 0.0,       # aderência baixa (hard zero)
                'diagnostico_estrutural': diagnostico_estrutural,
                'titulo': titulo,
                'texto': texto,
                'aderencia_cliente': aderencia_cliente
            }), 200

        # ======= Correção normal (temas livres) =======
        # Aderência entre tema informado e (título + texto)
        aderencia = theme_adherence_score(texto, tema, titulo)  # 0..1

        notas = {
            "competencia_1": int(analisar_competencia_1(texto)),
            "competencia_2": int(analisar_competencia_2(texto, tema, titulo)),  # usa aderência considerando título
            "competencia_3": int(analisar_competencia_3(texto)),
            "competencia_4": int(analisar_competencia_4(texto)),
            "competencia_5": int(analisar_competencia_5(texto))                 # sem penalidade de alinhamento
        }
        nota_final = float(calcular_nota_final(notas))

        # calibração dos extremos
        nota_final *= fator_severidade(texto, notas)
        nota_final *= boost_excelente(notas)
        nota_final = curva_topo_suave(nota_final)
        # bônus ouro opcional
        if min(notas.values()) >= 160 and nota_final >= 860:
            nota_final = min(1000.0, nota_final + 35.0)
        nota_final = round(max(0.0, min(1000.0, nota_final)), 2)

        comentarios = gerar_comentarios(notas, tema)

        diagnostico_estrutural = {
            'palavras': len(re.findall(r"\w+", normalize_text_basic(texto))),
            'paragrafos': len([p for p in texto.split('\n') if p.strip()]),
            'frases': len(safe_sent_tokenize(texto))
        }

        # Detector auxiliar (não afeta nota) – útil para telemetria/debug
        tema_aux, score_aux = detect_theme_tfidf(texto)

        return jsonify({
            'versao_corretor': VERSAO_CORRETOR,
            'gerado_em': datetime.utcnow().isoformat() + 'Z',
            'nota_final': nota_final,
            'notas': notas,
            'comentarios': comentarios,
            'criterios': CRITERIOS,
            'tema_informado': tema,
            'tema_detectado': tema,                     # compatível com o front
            'score_tema': round(float(aderencia), 4),   # aderência (0..1) tema vs (título+texto)
            'diagnostico_estrutural': diagnostico_estrutural,
            'titulo': titulo,                           # ecoa o título recebido
            'texto': texto,
            'aderencia_cliente': aderencia_cliente,     # ecoa heurística do front (se vier)
            'debug_tema_aux': {                         # campo informativo (não usar no front)
                'tema_aux': tema_aux,
                'score_aux': round(float(score_aux or 0.0), 4)
            }
        }), 200

    except Exception as e:
        print("[ERRO] /corrigir-redacao:", e)
        traceback.print_exc()
        return jsonify({'error': 'Erro interno no servidor.', 'detalhes': str(e)}), 500



# =========================
#  STARTUP
# =========================
if __name__ == '__main__':
    try:
        build_theme_vectorizer()  # opcional/diagnóstico
    except Exception as e:
        print("[AVISO] Falha ao construir vetores auxiliares:", e)
    print("URL map:", app.url_map)
    # produção: usar WSGI (gunicorn/uwsgi) e debug=False
    app.run(debug=True)
