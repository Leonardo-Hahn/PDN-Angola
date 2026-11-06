#!/usr/bin/env python
"""
03_ngramas.py

Conta monogramas, bigramas e trigramas (1, 2 e 3 palavras vizinhas) em cada plano.

Entrada : data/processed/paragrafos.parquet
Saídas  : outputs/tables/monogramas_<documento>.csv  (todas as palavras do documento)
          outputs/tables/bigramas_<documento>.csv    (todos os bigramas do documento)
          outputs/tables/trigramas_<documento>.csv   (todos os trigramas do documento)
          outputs/tables/monogramas_top.csv          (os N mais frequentes de cada plano, lado a lado)
          outputs/tables/bigramas_top.csv
          outputs/tables/trigramas_top.csv

Como conta:
  - só parágrafos de texto (títulos ficam de fora);
  - minúsculas; palavra = sequência de letras, com hífen interno ("bem-estar");
    números e siglas com dígitos saem;
  - o n-grama é formado com as palavras VIZINHAS no texto e só depois se filtram
    as stopwords. Assim "luta contra a pobreza" não vira o bigrama artificial
    "luta pobreza";
  - monograma: palavra que não é stopword;
  - bigrama: nenhuma das duas palavras pode ser stopword ("ensino superior");
  - trigrama: as pontas não podem ser stopword, o meio pode. Em português o meio
    costuma ser preposição: "plano de desenvolvimento", "combate à pobreza";
  - n-gramas não atravessam parágrafos;
  - por_10mil = ocorrências por 10 mil n-gramas do documento, para comparar planos
    de tamanhos diferentes.

Uso (da raiz do projeto):
    uv run python scripts/03_ngramas.py
    uv run python scripts/03_ngramas.py --top 30
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Stopwords do português (artigos, preposições, contrações, pronomes, conjunções,
# verbos auxiliares), com as grafias pré-AO90 que aparecem nos planos.
STOPWORDS = set(
    """
    a à às ao aos o os as um uma uns umas
    de do da dos das d em no na nos nas num numa por pelo pela pelos pelas
    para pra com sem sob sobre entre até após ante contra desde perante durante
    e ou nem mas porém contudo todavia que se como quando onde enquanto porque pois
    também já não sim mais menos muito muita muitos muitas pouco pouca tão tanto
    me te lhe lhes nos vos ele ela eles elas eu tu nós vós
    seu sua seus suas meu minha nosso nossa nossos nossas
    este esta estes estas esse essa esses essas aquele aquela aqueles aquelas
    isto isso aquilo qual quais cujo cuja cujos cujas quem
    todo toda todos todas outro outra outros outras mesmo mesma mesmos mesmas
    cada qualquer quaisquer vários várias algum alguma alguns algumas nenhum nenhuma
    ser é são era eram foi foram será serão seja sejam sendo sido
    estar está estão esteve estiveram estará estarão
    ter tem têm tinha teve tiveram terá terão haver há houve haverá
    pode podem poderá poderão deve devem deverá deverão
    assim ainda bem só apenas cerca nomeadamente designadamente
    """.split()
)
PALAVRA = re.compile(r"[a-zà-öø-ÿ]+(?:-[a-zà-öø-ÿ]+)*")
NOMES = {1: "monogramas", 2: "bigramas", 3: "trigramas"}


def tokens(texto: str) -> list:
    return PALAVRA.findall(texto.lower())


def ngramas(texto: str, n: int):
    t = tokens(texto)
    for i in range(len(t) - n + 1):
        g = t[i : i + n]
        if any(len(w) < 2 for w in g):
            continue
        # só as pontas são testadas: no bigrama isso cobre as duas palavras;
        # no trigrama o meio pode ser preposição ("combate à pobreza")
        if g[0] in STOPWORDS or g[-1] in STOPWORDS:
            continue
        yield " ".join(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="data/processed/paragrafos.parquet")
    ap.add_argument("--tabelas", default="outputs/tables")
    ap.add_argument("--top", type=int, default=20, help="quantos n-gramas mostrar por documento")
    args = ap.parse_args()

    par = pd.read_parquet(args.entrada)
    par = par[~par["titulo"]]
    tab = Path(args.tabelas)
    tab.mkdir(parents=True, exist_ok=True)

    for n, nome in NOMES.items():
        tops = {}
        for doc, g in par.groupby("doc", sort=True):
            cont = Counter(x for t in g["texto"] for x in ngramas(t, n))
            total = sum(cont.values())
            df = pd.DataFrame(cont.items(), columns=[nome[:-1], "n"])
            df["por_10mil"] = (df["n"] / total * 10_000).round(1)
            df = df.sort_values(["n", nome[:-1]], ascending=[False, True]).reset_index(drop=True)
            df.to_csv(tab / f"{nome}_{doc}.csv", index=False)
            tops[doc] = df.head(args.top)
            print(f"\n{nome.upper()} — {doc}: {total} ({len(df)} distintos)")
            print(df.head(args.top).to_string(index=False))

        lado = pd.concat(
            {doc: t[[nome[:-1], "n"]].reset_index(drop=True) for doc, t in tops.items()}, axis=1
        )
        lado.index = range(1, len(lado) + 1)
        lado.to_csv(tab / f"{nome}_top.csv", index_label="posicao")

    print(f"\nTabelas em {tab}/ (monogramas_*.csv, bigramas_*.csv, trigramas_*.csv)")


if __name__ == "__main__":
    main()
