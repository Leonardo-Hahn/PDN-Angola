#!/usr/bin/env python
"""
02_limpar.py

Limpa os blocos brutos e produz parágrafos prontos para n-gramas e modelagem de tópicos.

Entrada : data/processed/blocos_raw.parquet
Saídas  : data/processed/paragrafos.parquet      (o corpus limpo)
          data/processed/descartados.parquet     (tudo o que saiu, com o motivo)
          outputs/tables/resumo_limpeza.csv      (contagens por documento e motivo)
          outputs/tables/revisao_*.csv           (listas para conferência manual)

Antes das regras, os blocos de cada página são postos em ordem de leitura
(coluna da esquerda, depois a da direita) e os caracteres de controle usados
como marcador/tabulação no PDF são normalizados.

Regras por BLOCO (o primeiro motivo que casa é o registrado):
  1. fora_do_corpo     antes da introdução ou depois do fim definido em CONFIG
  2. numero_pagina     bloco só com número (arábico ou romano) na margem
  3. cabecalho_rodape  texto curto que se repete nas margens de várias páginas (dígitos ignorados)
  4. fonte_pequena     fonte < 85% da do corpo: tabelas, mapas, rótulos de gráfico, fontes
                       (exceto prosa: células de quadro e notas de rodapé com frases ficam)
  5. sumario           linhas com pontilhado "......  12"
  6. siglas            lista de abreviaturas ("MAT - Ministério ...")
  7. legenda           "Quadro 3", "Gráfico 2", "Figura 1", "Fonte:", "Nota:"
  8. numerico          resíduo de tabela (maioria de números/símbolos)
Regras por ITEM (linhas de um bloco já juntadas; marcador/numeração abre item novo):
  9. tabela            linha de tabela: termina em 3+ valores numéricos ("n.d." incluso)
 10. lista_siglas      lista de siglas separadas por vírgula ("MED, MESCTI, Governos Provinciais")
Regras por PARÁGRAFO (depois de juntar continuações entre blocos/páginas):
 11. rotulo_repetido   texto curto (<= 8 palavras), sem ponto final, repetido 3+ vezes no documento
 13. duplicado        parágrafo idêntico (dígitos ignorados) repetido 3+ vezes: fica só o 1º
 12. curto             menos de 4 palavras (título: sem nenhuma palavra de 4+ letras)

Desidifenização: "desen-/volvimento" vira "desenvolvimento", mas compostos
("bem-/estar", "destacando-/se", "mão-de-/obra") mantêm o hífen. A decisão usa
o vocabulário dos próprios PDFs (ver decide_hifen) e vai para revisao_hifens.csv.

Uso (da raiz do projeto):
    uv run python scripts/02_limpar.py
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG: ajuste aqui depois de olhar o resumo. Páginas contadas a partir de 1,
# como no leitor de PDF. "inicio" = primeira página do corpo; "fim" = última.
# Se um documento não estiver aqui (ou o valor for None), o script tenta achar
# sozinho e IMPRIME o que escolheu, para você conferir.
# Onde começam os anexos (página do PDF): 2013-2017 p. 215; 2018-2022 p. 303;
# 2023-2027 p. 196 (Anexo A) e p. 209 (Anexo B).
# ----------------------------------------------------------------------------
CONFIG = {
    "PDN_2013-2017":        {"inicio": 13, "fim": 234},
    "PDN_2018-2022":   {"inicio": 21, "fim": 306},
    "PDN_2023-2027":        {"inicio": 7, "fim": 209},
}

# Padronização de variantes (aplicada ao texto final, case-sensitive)
SUBSTITUICOES = [
    (r"\b[Ii]nfra-estrutura", lambda m: m.group(0).replace("-", "")),
    (r"\bPND\b", "PDN"),
]

MARGEM = 0.10          # fração superior/inferior da página tratada como margem
MIN_REPETICOES = 3     # um texto de margem precisa aparecer em pelo menos 3 páginas
MAX_PALAVRAS_CAB = 12  # cabeçalho/rodapé repetido tem no máximo isto de palavras
FONTE_PEQUENA = 0.85   # fração do tamanho de fonte do corpo abaixo da qual o bloco sai
TERMINAL = tuple('.!?:;»”)"…')
LETRA = "A-Za-zÀ-ÖØ-öø-ÿ"
BULLET = re.compile(r"^\s*[•▪●■◦·–—−\-]\s+")
TRACO = tuple("–—−-")  # só abrem item se a linha anterior fechou a frase
# numeração no início da linha: "7.", "4.2.2.", "a)", "iii)", "(2)"
ENUMERADOR = re.compile(r"^\s*\(?(\d{1,3}(\.\d{1,3})*[.)]|[a-z]\)|[ivxl]{1,5}[.)])(\s+|$)")
ROMANO = r"[ivxlcdm]+"
ENUNCIADO = re.compile(r"(?i:prioridade|objectivo|objetivo|meta)\s+\d+(\.\d+)*\s*:")
# palavras que, no fim de uma linha/parágrafo, indicam que a frase continua
LIGACAO = {
    "a", "à", "ao", "aos", "as", "às", "o", "os", "e", "ou", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para", "com",
    "que", "um", "uma", "uns", "umas", "sua", "seu", "suas", "seus", "entre", "sobre", "sem",
}
# pronomes átonos: "destacando-/se", "garantir-/lhes", "procurar-/se-á"
CLITICOS = {"se", "lhe", "lhes", "o", "a", "os", "as", "lo", "la", "los", "las",
            "no", "na", "nos", "nas", "me", "te", "vos"}
TOKEN = re.compile(rf"[{LETRA}]+(?:-[{LETRA}]+)*")


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def normaliza_chave(t: str) -> str:
    """Chave para comparar textos repetidos: minúsculas, dígitos viram '#'."""
    t = re.sub(r"\d+", "#", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def n_palavras(t: str) -> int:
    return len(re.findall(r"\w+", t))


def prepara_texto(t: str) -> str:
    """Normaliza caracteres do PDF mantendo as quebras de linha.
    NFC (e não NFKC) para não transformar 'n.º' em 'n.o'."""
    t = unicodedata.normalize("NFC", t)
    for a, b in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        t = t.replace(a, b)
    linhas = []
    for lin in t.split("\n"):
        # \x07, tab, espaços largos e \x03 aparecem como separadores dentro da linha
        lin = re.sub(r"[\x03\x07\t\u00a0\u2000-\u200a\u202f]", " ", lin)
        # \x01 e caracteres de uso privado (símbolos Wingdings) são marcadores de lista
        lin = "".join("• " if (c == "\x01" or unicodedata.category(c) == "Co") else c for c in lin)
        lin = re.sub(r"[\x00-\x08\x0b-\x1f]", "", lin)
        linhas.append(re.sub(r" {2,}", " ", lin).rstrip())
    return "\n".join(linhas)


def eh_numero_pagina(t: str) -> bool:
    t = t.strip()
    return bool(re.fullmatch(rf"(\d{{1,4}}|{ROMANO})", t, flags=re.I)) or bool(
        re.fullmatch(r"(p[áa]g(ina)?\.?\s*)?\d{1,4}\s*(/|de)\s*\d{1,4}", t, flags=re.I)
    )


def eh_sumario(t: str) -> bool:
    linhas = [l for l in t.split("\n") if l.strip()]
    if not linhas:
        return False
    com_ponto = sum(bool(re.search(r"(\.\s*){4,}\s*\d*\s*$|\.{4,}", l)) for l in linhas)
    return com_ponto / len(linhas) >= 0.3


def eh_siglas(t: str) -> bool:
    linhas = [l for l in t.split("\n") if l.strip()]
    if len(linhas) < 2:
        return False
    padrao = re.compile(r"^\s*[A-ZÀ-Ý][A-Za-zÀ-ÿ0-9&/.\-]{1,14}\s*[-–:]\s+\S")
    return sum(bool(padrao.match(l)) for l in linhas) / len(linhas) >= 0.6


def eh_legenda(t: str) -> bool:
    t0 = t.strip()
    # o identificador ("3", "2.1", "A", "IV") é sensível a maiúsculas, para não pegar
    # frases como "quadro macroeconómico do PDN ..."
    return bool(
        re.match(r"(?i:quadro|gr[áa]fico|figura|tabela|mapa|caixa)\s+(n\.?º\s*)?(\d|[A-Z]\b|[IVXL]+\b)", t0)
        or re.match(r"^(fonte|fontes|nota|notas)\s*:", t0, flags=re.I)
    )


def eh_numerico(t: str) -> bool:
    nao_esp = re.sub(r"\s+", "", t)
    if not nao_esp:
        return True
    letras = sum(c.isalpha() for c in nao_esp)
    if letras / len(nao_esp) < 0.5:
        return True
    toks = t.split()
    num = sum(bool(re.fullmatch(r"[\d.,%()\-–/+]+", x)) for x in toks)
    return len(toks) >= 3 and num / len(toks) >= 0.5


VALOR = re.compile(r"[\d.,%()\-–/+*‰~]*\d[\d.,%()\-–/+*‰~]*|n\.?\s?d\.?|n/d|[-–—]", flags=re.I)


def eh_linha_tabela(t: str) -> bool:
    """Linha de tabela: termina em 3 ou mais valores ('12,5 13,0 n.d.')."""
    toks = t.split()
    k = 0
    for x in reversed(toks):
        if VALOR.fullmatch(x):
            k += 1
        else:
            break
    return k >= 3


def eh_lista_siglas(t: str) -> bool:
    partes = [p.strip(" .;") for p in t.split(",") if p.strip(" .;")]
    if len(partes) < 2:
        return False
    sig = sum(bool(re.fullmatch(r"[A-ZÀ-Ý][A-ZÀ-Ý0-9&/\-]+", p)) for p in partes)
    return sig / len(partes) >= 0.5


def eh_prosa(t: str) -> bool:
    """Frase corrida (e não rótulo de tabela/gráfico): 12+ tokens, maioria de palavras
    em minúscula, quase sem números e com alguma pontuação de frase."""
    toks = t.split()
    if len(toks) < 12 or not re.search(r"[.;:?]", t):
        return False
    minusc = sum(bool(re.fullmatch(r"[(“\"]?[a-zà-ÿ][a-zà-ÿ\-]*[.,;:)”\"]?", x)) for x in toks)
    nums = sum(bool(re.search(r"\d", x)) for x in toks)
    return minusc / len(toks) >= 0.6 and nums / len(toks) < 0.1


def limpa_unicode(t: str) -> str:
    """Espaços simples; tira hífens suaves, menos o final (ainda pode ser juntado)."""
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\xad(?!$)", "", t)


# ----------------------------------------------------------------------------
# desidifenização
# ----------------------------------------------------------------------------
def monta_vocabulario(textos) -> Counter:
    """Conta as palavras dos PDFs que NÃO estão partidas no fim da linha.
    Palavras com hífen interno ('bem-estar') entram inteiras."""
    vocab: Counter = Counter()
    for t in textos:
        partida_antes = False
        for lin in prepara_texto(t).split("\n"):
            toks = TOKEN.findall(lin.lower())
            partida_agora = bool(re.search(rf"[{LETRA}][-\xad]$", lin))
            if partida_antes and toks:
                toks = toks[1:]
            if partida_agora and toks:
                toks = toks[:-1]
            vocab.update(toks)
            partida_antes = partida_agora
    return vocab


def partes_vocab(vocab: Counter) -> Counter:
    """Palavras simples, contando também as partes de compostos com hífen."""
    p: Counter = Counter()
    for w, n in vocab.items():
        for parte in w.split("-"):
            p[parte] += n
    return p


def decide_hifen(esq: str, dir_: str, vocab: Counter, simples: Counter) -> bool:
    """True = manter o hífen ('bem-estar'); False = juntar ('desenvolvimento').
    esq = pedaço antes do hífen; dir_ = palavra seguinte (pode ter hífen: 'se-á')."""
    a = esq.lower()
    b = TOKEN.match(dir_.lower())
    if not b:
        return False
    b = b.group(0)
    junto, hifen = vocab[a + b], vocab[a + "-" + b]
    if hifen > junto:
        return True
    if junto > 0:
        return False
    if b.split("-")[0] in CLITICOS:
        return True
    # as duas partes são palavras que existem sozinhas: composto
    return len(a) >= 2 and len(b) >= 2 and simples[a] > 0 and simples[b.split("-")[0]] > 0


class Hifenizador:
    def __init__(self, vocab: Counter):
        self.vocab = vocab
        self.simples = partes_vocab(vocab)
        self.log: Counter = Counter()

    def junta(self, atual: str, prox: str) -> str:
        """Junta 'atual' (terminado em hífen ou hífen suave) com 'prox'."""
        m = re.search(rf"([{LETRA}]+)([-\xad])$", atual)
        esq = m.group(1)
        prim = prox.split()[0] if prox.split() else ""
        if m.group(2) == "\xad":  # hífen suave: sempre quebra de sílaba
            manter = False
        elif re.match(r"[A-ZÀ-Ý]", prox):  # "Sul-/Sul", "ex-/Militares"
            manter = True
        else:
            manter = decide_hifen(esq, prim, self.vocab, self.simples)
        palavra = esq + ("-" if manter else "") + re.sub(r"[^\w\-]+$", "", prim)
        self.log[(palavra, "manteve" if manter else "juntou")] += 1
        return atual[:-1] + ("-" if manter else "") + prox

    def hifen_solto(self, t: str) -> str:
        """'infra- estruturas' dentro da linha (quebra antiga do PDF)."""
        def troca(m):
            prox = t[m.end():].split()[0]
            manter = decide_hifen(m.group(1), prox, self.vocab, self.simples)
            self.log[(m.group(1) + ("-" if manter else "") + prox, "manteve" if manter else "juntou")] += 1
            return m.group(1) + ("-" if manter else "")

        return re.sub(rf"([{LETRA}]+)- (?=[a-zà-ÿ])", troca, t)


def termina_partida(t: str) -> bool:
    return bool(re.search(rf"[{LETRA}][-\xad]$", t))


def monta_itens(texto: str, hif: Hifenizador):
    """Linhas de um bloco -> lista de itens (marcador ou numeração abre item novo).
    Junta linhas e trata o hífen de quebra ('desenvol-\\nvimento')."""
    itens = []  # (texto, abertura): abertura = "" | "marcador" (•, –) | "numero" (7., a), iii))
    atual, bullet = None, ""
    for lin in texto.split("\n"):
        lin = lin.strip()
        if not lin:
            continue
        m_bul = BULLET.match(lin)
        m_enum = ENUMERADOR.match(lin)
        # numeração só abre item se vier seguida de maiúscula ou sozinha na linha
        if m_enum and not re.match(r"$|[A-ZÀ-Ý“\"«]", lin[m_enum.end():]):
            m_enum = None
        if m_bul and lin.lstrip().startswith(TRACO) and atual and not atual.endswith((".", ";", ":", "!", "?")):
            m_bul = None  # travessão no meio da frase: "... Trabalho)\n– visando ..."
        if m_bul or m_enum:
            if atual:
                itens.append((atual, bullet))
            atual = lin[(m_bul or m_enum).end():].strip()
            bullet = "marcador" if m_bul else "numero"
            continue
        if atual is None:
            atual, bullet = lin, ""
        elif atual == "":  # numeração sozinha na linha anterior ("7.")
            atual = lin
        elif termina_partida(atual):
            atual = hif.junta(atual, lin)
        else:
            atual += " " + lin
    if atual:
        itens.append((atual, bullet))
    return itens


# ----------------------------------------------------------------------------
# ordem de leitura
# ----------------------------------------------------------------------------
def junta_girados(pg: pd.DataFrame) -> pd.DataFrame:
    """Tabelas em paisagem (texto girado 90°): o pymupdf devolve cada linha como
    um bloco. Agrupa linhas vizinhas e alinhadas em células e passa as coordenadas
    para o referencial do texto (x = sentido da leitura). As células saem em ordem
    de coluna: a coluna inteira de cima para baixo, depois a seguinte."""
    gir = pg[pg["girado"] > 0.5].sort_values("x0")
    if gir.empty:
        return pg
    celulas = []  # listas de índices
    for i, r in gir.iterrows():
        alvo = None
        for c in celulas:
            u = gir.loc[c[-1]]
            alt_linha = u["x1"] - u["x0"]
            sobrepoe = min(r["y1"], u["y1"]) - max(r["y0"], u["y0"]) > 0
            if 0 <= r["x0"] - u["x1"] <= alt_linha and sobrepoe and abs(r["y1"] - u["y1"]) < 3 * alt_linha:
                alvo = c
        if alvo is None:
            celulas.append([i])
        else:
            alvo.append(i)
    novas = []
    for c in celulas:
        g = gir.loc[c]
        base = g.iloc[0].to_dict()
        larg, alt = base["larg_pag"], base["alt_pag"]
        base.update(
            x0=alt - g["y1"].max(), x1=alt - g["y0"].min(), y0=g["x0"].min(), y1=g["x1"].max(),
            larg_pag=alt, alt_pag=larg, tam_fonte=g["tam_fonte"].mean(), negrito=g["negrito"].mean(),
            n_linhas=int(g["n_linhas"].sum()), texto="\n".join(g["texto"]),
        )
        novas.append(base)
    return pd.concat([pg[pg["girado"] <= 0.5], pd.DataFrame(novas)], ignore_index=True)


def ordem_leitura(pg: pd.DataFrame) -> list:
    """Índices dos blocos de uma página em ordem de leitura.
    Blocos que atravessam o meio da página separam faixas; dentro de cada faixa
    lê-se a coluna da esquerda inteira e depois a da direita."""
    if pg.empty:
        return []
    meio = float(pg["larg_pag"].iloc[0]) / 2
    tol = 20
    ordem, esq, dir_ = [], [], []

    def descarrega():
        ordem.extend(sorted(esq, key=lambda i: pg.at[i, "y0"]))
        ordem.extend(sorted(dir_, key=lambda i: pg.at[i, "y0"]))
        esq.clear()
        dir_.clear()

    for i in pg.sort_values(["y0", "x0"]).index:
        x0, x1 = pg.at[i, "x0"], pg.at[i, "x1"]
        if x1 <= meio + tol:
            esq.append(i)
        elif x0 >= meio - tol:
            dir_.append(i)
        else:
            descarrega()
            ordem.append(i)
    descarrega()
    return ordem


# ----------------------------------------------------------------------------
# corpo do documento (início / fim)
# ----------------------------------------------------------------------------
def acha_corpo(d: pd.DataFrame, stem: str):
    n_pag = int(d["pagina"].max())
    cfg = CONFIG.get(stem, {})
    inicio, fim = cfg.get("inicio"), cfg.get("fim")
    origem = {"inicio": "CONFIG", "fim": "CONFIG"}

    if inicio is None:
        origem["inicio"] = "auto"
        inicio = 1
        for _, r in d.sort_values(["pagina", "bloco"]).iterrows():
            t = r["texto"].strip()
            if (
                n_palavras(t) <= 6
                and re.match(r"^(\d+[.)]?\s*)?(introdu[cç][aã]o|apresenta[cç][aã]o|sum[aá]rio executivo)\b", t, flags=re.I)
                and ".." not in t
                and r["pagina"] <= 0.5 * n_pag
            ):
                inicio = int(r["pagina"])
                break
        else:
            origem["inicio"] = "auto (NÃO achou introdução; usando página 1)"

    if fim is None:
        origem["fim"] = "auto"
        fim = n_pag
        cand = d[(d["pagina"] > max(inicio, 0.4 * n_pag))].sort_values(["pagina", "bloco"])
        for _, r in cand.iterrows():
            t = r["texto"].strip()
            if (
                n_palavras(t) <= 10
                and re.match(r"^anexos?\b", t, flags=re.I)
                and ".." not in t
                and not t.rstrip().endswith(".")
            ):
                fim = int(r["pagina"]) - 1
                break
        else:
            origem["fim"] = "auto (NÃO achou anexos; usando última página)"
    return int(inicio), int(fim), origem


# ----------------------------------------------------------------------------
# pipeline por documento
# ----------------------------------------------------------------------------
def processa_doc(d: pd.DataFrame, stem: str, hif: Hifenizador):
    d = d.sort_values(["pagina", "bloco"]).reset_index(drop=True)
    d["texto"] = d["texto"].map(prepara_texto)
    d = d[d["texto"].str.strip() != ""]
    paginas = []
    for _, pg in d.groupby("pagina", sort=True):
        pg = junta_girados(pg)
        normais, gir = pg[pg["girado"] <= 0.5], pg[pg["girado"] > 0.5]
        gir = gir.assign(coluna=(gir["x0"] / 20).round()).sort_values(["coluna", "y0"])
        paginas.append(pg.loc[ordem_leitura(normais) + list(gir.index)])
    d = pd.concat(paginas, ignore_index=True)

    n_pag = int(d["pagina"].max())
    inicio, fim, origem = acha_corpo(d, stem)
    print(f"  {stem}: corpo = páginas {inicio}–{fim}  [início: {origem['inicio']}; fim: {origem['fim']}]")

    # tamanho de fonte do corpo = mediana dos blocos longos dentro do corpo
    no_corpo = d["pagina"].between(inicio, fim)
    longos = d[no_corpo & (d["texto"].map(n_palavras) >= 20)]
    corpo = float(longos["tam_fonte"].median()) if len(longos) else float(d["tam_fonte"].median())

    # textos curtos repetidos nas margens
    em_margem = (d["y1"] <= MARGEM * d["alt_pag"]) | (d["y0"] >= (1 - MARGEM) * d["alt_pag"])
    d["chave"] = d["texto"].map(normaliza_chave)
    curto_margem = em_margem & (d["texto"].map(n_palavras) <= MAX_PALAVRAS_CAB)
    paginas_por_chave = d[curto_margem].groupby("chave")["pagina"].nunique()
    chaves_cab = set(paginas_por_chave[paginas_por_chave >= MIN_REPETICOES].index)

    motivos = []
    for i, r in d.iterrows():
        t, mot = r["texto"], None
        if r["pagina"] < inicio or r["pagina"] > fim:
            mot = "fora_do_corpo"
        elif em_margem.iloc[i] and eh_numero_pagina(t):
            mot = "numero_pagina"
        elif curto_margem.iloc[i] and r["chave"] in chaves_cab:
            mot = "cabecalho_rodape"
        elif r["tam_fonte"] < FONTE_PEQUENA * corpo and not eh_prosa(t.replace("\n", " ")):
            mot = "fonte_pequena"
        elif eh_sumario(t):
            mot = "sumario"
        elif eh_siglas(t):
            mot = "siglas"
        elif eh_legenda(t):
            mot = "legenda"
        elif eh_numerico(t):
            mot = "numerico"
        motivos.append(mot)
    d["motivo"] = motivos
    # prosa em fonte pequena que fica (células de quadro, notas de rodapé):
    # tira o número de chamada colado ("5Embora o Plano ...")
    peq = d["motivo"].isna() & (d["tam_fonte"] < FONTE_PEQUENA * corpo)
    d.loc[peq, "texto"] = d.loc[peq, "texto"].str.replace(r"^\s*\d{1,2}(?=[A-ZÀ-Ý][a-zà-ÿ])", "", regex=True)

    descartados = d[d["motivo"].notna()][["doc", "pagina", "motivo", "texto"]].copy()
    mantidos = d[d["motivo"].isna()]
    desc_itens = []

    # monta parágrafos
    pars = []  # dicts
    secao, secao_ant = "", ""
    for _, r in mantidos.iterrows():
        for texto, bullet in monta_itens(r["texto"], hif):
            texto = limpa_unicode(texto)
            if not texto:
                continue
            if eh_linha_tabela(texto) or eh_lista_siglas(texto):
                mot = "tabela" if eh_linha_tabela(texto) else "lista_siglas"
                desc_itens.append({"doc": stem, "pagina": int(r["pagina"]), "motivo": mot, "texto": texto})
                continue
            palavras = n_palavras(texto)
            # cabeçalho numerado ("4.4. Riscos ...") pode ser título; item com marcador não.
            # "Prioridade 3.2.1: ...", "Objectivo 6.3: ...", "Meta 1.1: ..." são enunciados
            # de política em negrito: conteúdo, não título.
            titulo = (
                palavras <= 15
                and not texto.endswith((".", ";", ","))
                and (r["negrito"] >= 0.6 or r["tam_fonte"] >= corpo * 1.1)
                and bullet != "marcador"
                and not ENUNCIADO.match(texto)
            )
            ant = pars[-1] if pars else None
            continua = False
            if ant and not bullet and re.match(r"[a-zà-ÿ(]", texto):
                # minúscula: continua o último parágrafo inacabado. Olha até 3 para
                # trás, porque caixas de destaque e títulos de coluna podem ficar no meio;
                # o pedaço anterior pode ter sido tomado por título (negrito) por engano.
                for k in range(len(pars) - 1, max(len(pars) - 4, -1), -1):
                    if pars[k]["titulo"] and k < len(pars) - 1:
                        continue
                    if not pars[k]["texto"].endswith(TERMINAL) and pars[k]["pagina"] >= r["pagina"] - 1:
                        ant, continua = pars[k], True
                        break
            elif ant and not bullet and not titulo and not ant["titulo"] and not ant["texto"].endswith(TERMINAL):
                # maiúscula depois de "... de" / "... para o": mesma frase ("do\nPlano")
                ult = re.findall(r"\w+$", ant["texto"].lower())
                continua = bool(ult and ult[0] in LIGACAO)
            if continua:
                if ant["titulo"]:
                    ant["titulo"] = False
                    secao = secao_ant
                if termina_partida(ant["texto"]):
                    ant["texto"] = hif.junta(ant["texto"], texto)
                else:
                    ant["texto"] += " " + texto
                continue
            if titulo:
                secao_ant, secao = secao, texto
            pars.append({"doc": stem, "pagina": int(r["pagina"]), "titulo": titulo, "secao": secao, "texto": texto})

    # regras 11-12 e normalizações finais
    for p in pars:
        p["texto"] = hif.hifen_solto(limpa_unicode(p["texto"]).replace("\xad", ""))
        # chamada de nota colada à palavra ("privados5 e", "habitantes2."); seguida de
        # ":" é numeração sem espaço ("Objectivo1: Impulsionar")
        p["texto"] = re.sub(
            rf"\b([{LETRA}]*[a-zà-ÿ]{{3}})(\d{{1,2}})\b(:?)",
            lambda m: m.group(1) + (" " + m.group(2) + ":" if m.group(3) else ""),
            p["texto"],
        )
        for pad, rep in SUBSTITUICOES:
            p["texto"] = re.sub(pad, rep, p["texto"])
        p["n_palavras"] = n_palavras(p["texto"])
    rep = Counter(normaliza_chave(p["texto"]) for p in pars if not p["titulo"])
    finais, vistos = [], set()
    for p in pars:
        mot, chave = None, normaliza_chave(p["texto"])
        if p["titulo"] and not re.search(rf"[{LETRA}]{{4,}}", p["texto"]):
            mot = "curto"  # "O", "Top", "E A": letras soltas de mapa ou gráfico
        elif not p["titulo"]:
            if p["n_palavras"] <= 8 and not p["texto"].endswith(".") and rep[normaliza_chave(p["texto"])] >= 3:
                mot = "rotulo_repetido"
            elif p["n_palavras"] < 4:
                mot = "curto"
            elif rep[chave] >= 3 and chave in vistos:
                mot = "duplicado"  # frase-modelo repetida (infográfico, fichas de província)
            vistos.add(chave)
        if mot:
            desc_itens.append({"doc": stem, "pagina": p["pagina"], "motivo": mot, "texto": p["texto"]})
        else:
            finais.append(p)
    for i, p in enumerate(finais, start=1):
        p["par_id"] = i

    desc = pd.concat([descartados, pd.DataFrame(desc_itens)], ignore_index=True)
    return finais, desc, {"inicio": inicio, "fim": fim, "paginas": n_pag}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="data/processed/blocos_raw.parquet")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--tabelas", default="outputs/tables")
    args = ap.parse_args()

    blocos = pd.read_parquet(args.entrada)
    proc, tab = Path(args.proc), Path(args.tabelas)
    proc.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    hif = Hifenizador(monta_vocabulario(blocos["texto"]))
    todos_p, todos_d, info = [], [], {}
    print("Documentos:")
    for stem, d in blocos.groupby("doc", sort=True):
        p, desc, inf = processa_doc(d.copy(), stem, hif)
        todos_p.extend(p)
        todos_d.append(desc)
        info[stem] = inf

    par = pd.DataFrame(todos_p)[["doc", "par_id", "pagina", "titulo", "secao", "n_palavras", "texto"]]
    desc = pd.concat(todos_d, ignore_index=True)
    par.to_parquet(proc / "paragrafos.parquet", index=False)
    desc.to_parquet(proc / "descartados.parquet", index=False)

    # resumo
    corpo_par = par[~par["titulo"]]
    linhas = []
    for stem in sorted(info):
        sub = corpo_par[corpo_par["doc"] == stem]
        dd = desc[desc["doc"] == stem]["motivo"].value_counts().to_dict()
        linhas.append(
            {
                "doc": stem,
                "paginas_pdf": info[stem]["paginas"],
                "corpo_de": info[stem]["inicio"],
                "corpo_ate": info[stem]["fim"],
                "paragrafos": len(sub),
                "titulos": int(par[(par["doc"] == stem) & par["titulo"]].shape[0]),
                "palavras": int(sub["n_palavras"].sum()),
                **{f"desc_{k}": v for k, v in sorted(dd.items())},
            }
        )
    resumo = pd.DataFrame(linhas).fillna(0)
    resumo.to_csv(tab / "resumo_limpeza.csv", index=False)

    # listas de revisão
    rep = (
        par.assign(chave=par["texto"].map(normaliza_chave))
        .groupby(["doc", "chave"])
        .agg(vezes=("texto", "size"), exemplo=("texto", "first"))
        .reset_index()
    )
    rep = rep[rep["vezes"] >= 3].sort_values(["vezes", "doc", "chave"], ascending=[False, True, True])
    rep.to_csv(tab / "revisao_repetidos.csv", index=False)

    pd.DataFrame(
        [(w, dec, n) for (w, dec), n in hif.log.items()], columns=["palavra", "decisao", "vezes"]
    ).sort_values(["decisao", "vezes", "palavra"], ascending=[False, False, True]).to_csv(
        tab / "revisao_hifens.csv", index=False
    )

    amostra = (
        desc.assign(texto=desc["texto"].str.replace("\n", " / ").str.slice(0, 200))
        .sample(frac=1, random_state=1)
        .groupby(["doc", "motivo"])
        .head(15)
        .sort_values(["doc", "motivo", "pagina"])
    )
    amostra.to_csv(tab / "revisao_descartados.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nResumo:")
    print(resumo.to_string(index=False))
    print(f"\nParágrafos: {proc/'paragrafos.parquet'}")
    print(f"Descartados: {proc/'descartados.parquet'}")
    print(f"Listas de revisão em {tab}/ (revisao_repetidos, revisao_hifens, revisao_descartados)")


if __name__ == "__main__":
    main()
