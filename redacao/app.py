# app.py
from flask import Flask, request, jsonify
import os
import re
import json
import math
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK (com fallbacks – não baixa nada em runtime)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)

# =========================
#  CONFIG / ARQUIVOS
# =========================
REDACOES_JSON_PATH = os.environ.get("REDACOES_JSON_PATH", "redacoes_exemplo.json")

def load_redacoes_exemplo():
	"""
	Carrega redacoes_exemplo.json (tema -> str ou [str]).
	Caso não exista, cria um fallback mínimo em memória.
	"""
	try:
		with open(REDACOES_JSON_PATH, "r", encoding="utf-8") as f:
			data = json.load(f)
			# sanity check
			if not isinstance(data, dict):
				raise ValueError("redacoes_exemplo.json inválido: raiz deve ser objeto.")
			return data
	except FileNotFoundError:
		# Fallback mínimo (você pode inserir mais exemplos via JSON externo)
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
#  CRITÉRIOS / TEMAS
# =========================
CRITERIOS = {
	"competencia_1": {"nome": "Domínio da norma padrão da língua escrita", "peso": 2.0, "descricao": "Avalia o domínio da norma culta da língua portuguesa."},
	"competencia_2": {"nome": "Compreensão da proposta de redação", "peso": 2.0, "descricao": "Avalia se o texto atende à proposta temática e ao tipo textual solicitado."},
	"competencia_3": {"nome": "Capacidade de organizar e relacionar informações", "peso": 2.0, "descricao": "Avalia a organização lógica do texto e a articulação entre as partes."},
	"competencia_4": {"nome": "Demonstração de conhecimento da língua necessária para argumentação", "peso": 2.0, "descricao": "Avalia o uso de recursos linguísticos para construir a argumentação."},
	"competencia_5": {"nome": "Elaboração de proposta de intervenção para o problema abordado", "peso": 2.0, "descricao": "Avalia a proposta de intervenção social para o problema discutido."}
}

TEMAS_REDACAO = {
	"educacao": ["educação", "escola", "professor", "ensino", "aprendizado", "aluno", "universidade"],
	"meio_ambiente": ["meio ambiente", "natureza", "sustentabilidade", "poluição", "desmatamento", "recursos naturais"],
	"saude": ["saúde", "hospital", "médico", "doença", "prevenção", "sus", "tratamento"],
	"violencia": ["violência", "segurança", "crime", "agressão", "homicídio", "polícia", "armas"],
	"tecnologia": ["tecnologia", "internet", "redes sociais", "digital", "inovação", "ciência", "robótica"],
	# Tema adicional do seu exemplo:
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
#  PREPROCESS
# =========================
def normalize_text_basic(text: str) -> str:
	text = text.lower()
	table = str.maketrans('', '', string.punctuation.replace('-', ''))
	return text.translate(table)

def preprocess_text_for_ttr(text: str) -> list:
	text_norm = normalize_text_basic(text)
	tokens = [t for t in re.findall(r"\w+", text_norm, flags=re.UNICODE) if t]
	return [t for t in tokens if t not in STOPWORDS_PT]

def preprocess_text_for_tfidf(text: str) -> str:
	return normalize_text_basic(text)

# =========================
#  TF-IDF – DETECÇÃO DE TEMA
# =========================
THEME_VECTOR_DATA = None

def build_theme_vectorizer():
	"""
	Monta o corpus com todas as redações do JSON + um pseudo-doc de palavras-chave por tema.
	Suporta: REDACOES_EXEMPLO[tema] como string ou lista de strings.
	"""
	global THEME_VECTOR_DATA
	temas = []
	corpus = []

	# 1) exemplos reais do JSON externo
	for tema, exemplos in REDACOES_EXEMPLO.items():
		if isinstance(exemplos, str):
			exemplos = [exemplos]
		for exemplo in exemplos:
			temas.append(tema)
			corpus.append(preprocess_text_for_tfidf(exemplo))

	# 2) palavras-chave como pseudo-documento (ajuda em temas com poucos exemplos)
	for tema, termos in TEMAS_REDACAO.items():
		temas.append(tema)
		corpus.append(preprocess_text_for_tfidf(" ".join(termos)))

	if not corpus:
		# fallback duro
		temas = ["outros"]
		corpus = ["texto geral"]
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(corpus)
	THEME_VECTOR_DATA = {"temas": temas, "vectorizer": vectorizer, "X": X}

def detect_theme_tfidf(texto: str) -> (str, float):
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
	if score < 0.08:
		return "outros", 0.0
	return tema_detectado, score

def detectar_tema(texto: str) -> str:
	tema, _ = detect_theme_tfidf(texto)
	if tema != "outros":
		return tema
	texto_limp = preprocess_text_for_tfidf(texto)
	counts = {k: sum(1 for term in v if term in texto_limp) for k, v in TEMAS_REDACAO.items()}
	if not counts:
		return "outros"
	best = max(counts, key=counts.get)
	return best if counts[best] > 0 else "outros"

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
#  COMPETÊNCIAS
# =========================
def analisar_competencia_1(texto: str) -> int:
	erros = 0
	paragrafos = [p for p in texto.split('\n') if p.strip()]
	if len(paragrafos) < 4:
		erros += (4 - len(paragrafos)) * 1.5
	frases = safe_sent_tokenize(texto)
	if len(frases) < 5:
		erros += 2
	palavras = normalize_text_basic(texto).split()
	for i in range(len(palavras)-1):
		if palavras[i] == "a" and palavras[i+1] in ["meninos", "homens"]:
			erros += 1
		if palavras[i] == "o" and palavras[i+1] in ["meninas", "mulheres"]:
			erros += 1
	erros = min(erros, 12)
	score = max(0.0, 1.0 - (erros / 12.0))
	return round(score * 200)

def analisar_competencia_2(texto: str, tema_informado: str) -> int:
	tema_detectado, score_tema = detect_theme_tfidf(texto)
	tema_ok = (tema_detectado == tema_informado)
	marcadores = [
		"em primeiro lugar","além disso","por outro lado","no entanto",
		"portanto","dessa forma","assim sendo","logo","todavia","entretanto"
	]
	texto_l = normalize_text_basic(texto)
	marcadores_presentes = sum(1 for m in marcadores if m in texto_l)
	s_tema = score_tema if tema_ok else score_tema * 0.3
	s_marc = min(1.0, marcadores_presentes / 5.0)
	score = 0.65 * s_tema + 0.35 * s_marc
	return max(0, min(200, round(score * 200)))

def analisar_competencia_3(texto: str) -> int:
	paragrafos = [p.strip() for p in texto.split('\n') if p.strip()]
	if len(paragrafos) < 3:
		return 80
	pars_proc = [preprocess_text_for_tfidf(p) for p in paragrafos]
	try:
		v = TfidfVectorizer()
		X = v.fit_transform(pars_proc)
	except ValueError:
		return 100
	sims = []
	for i in range(len(paragrafos)-1):
		s = cosine_similarity(X[i:i+1], X[i+1:i+2])[0][0]
		sims.append(s)
	avg_sim = float(np.mean(sims)) if sims else 0.0
	# repetição forte (mais tolerante: 0.98)
	repeticao_forte = False
	for i in range(len(paragrafos)):
		for j in range(i+1, len(paragrafos)):
			s = cosine_similarity(X[i:i+1], X[j:j+1])[0][0]
			if s > 0.98:
				repeticao_forte = True
				break
		if repeticao_forte:
			break
	if avg_sim <= 0.15:
		base = 0.35
	elif avg_sim >= 0.90:
		base = 0.30
	else:
		base = 0.55 + 0.40 * ((avg_sim - 0.15) / (0.90 - 0.15))
		base = max(0.0, min(0.95, base))
	if repeticao_forte:
		base -= 0.20
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
	# normalizar por 100 palavras (ok), mas com saturação moderada
	conectivos_por_100 = (qtd_conectivos / total_palavras) * 100.0
	# MTLD
	mtld_val = mtld(tokens_ttr)
	mtld_norm = max(0.0, min(1.0, mtld_val / 60.0))
	conn_norm = max(0.0, min(1.0, conectivos_por_100 / 4.0))  # satura em ~4/100
	score = 0.65 * mtld_norm + 0.35 * conn_norm
	return max(0, min(200, round(score * 200)))

def analisar_competencia_5(texto: str, tema: str) -> int:
	paragrafos = [p.strip() for p in texto.split('\n') if p.strip()]
	if not paragrafos:
		return 0
	ultimos = paragrafos[-2:] if len(paragrafos) >= 2 else [paragrafos[-1]]
	candidato = " ".join(ultimos).lower()
	palavras_chave = [
		"solução","medida","proposta","intervenção","ação",
		"sugere-se","é necessário","deve-se","precisa-se","implementação","política pública",
		"é fundamental que","propõe-se","recomenda-se","deve-se garantir"
	]
	tem_proposta = any(w in candidato for w in palavras_chave)
	if not tem_proposta:
		return 80  # mais brando: pode estar implícita
	tema_detectado, score_tema = detect_theme_tfidf(candidato)
	alinhado = (tema_detectado == tema) or (score_tema > 0.10)
	agentes = ["governo","estado","município","sociedade","escola","família","empresas","ong","população","comunidade"]
	qtd_agentes = sum(candidato.count(a) for a in agentes)
	detalhes_chave = ["por meio", "com ", "via ", "recursos", "orçamento", "programa", "plano", "prazo", "meta", "indicadores", "valorização"]
	tem_detalhes = any(d in candidato for d in detalhes_chave)
	s_base = 0.55 + (0.25 if alinhado else 0.0)
	s_ag = 0.2 if qtd_agentes >= 2 else (0.1 if qtd_agentes == 1 else 0.0)
	s_det = 0.2 if tem_detalhes else 0.0
	score = min(1.0, s_base + s_ag + s_det)
	return max(0, min(200, round(score * 200)))

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
	return round(score_norm, 2)

# =========================
#  ENDPOINT
# =========================
@app.route('/corrigir-redacao', methods=['POST'])
def corrigir_redacao():
	try:
		data = request.get_json()
		if not data or 'texto' not in data or 'tema' not in data:
			return jsonify({'error': 'Dados incompletos. Envie "texto" e "tema".'}), 400

		texto = data['texto']
		tema = data['tema']

		notas = {
			"competencia_1": analisar_competencia_1(texto),
			"competencia_2": analisar_competencia_2(texto, tema),
			"competencia_3": analisar_competencia_3(texto),
			"competencia_4": analisar_competencia_4(texto),
			"competencia_5": analisar_competencia_5(texto, tema)
		}
		nota_final = calcular_nota_final(notas)
		comentarios = gerar_comentarios(notas, tema)
		tema_detectado, score_tema = detect_theme_tfidf(texto)

		return jsonify({
			'nota_final': nota_final,
			'notas': notas,
			'comentarios': comentarios,
			'criterios': CRITERIOS,
			'tema_informado': tema,
			'tema_detectado': tema_detectado,
			'score_tema': round(float(score_tema), 4),
			'texto': texto
		})
	except Exception as e:
		return jsonify({'error': 'Erro interno no servidor.', 'detalhes': str(e)}), 500

if __name__ == '__main__':
	# produção: debug=False e WSGI (gunicorn/uwsgi)
	app.run(debug=True)
