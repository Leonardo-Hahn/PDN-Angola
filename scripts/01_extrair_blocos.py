#!/usr/bin/env python
"""
01_extrair_blocos.py

Transforma cada PDF de data/raw/ numa tabela de blocos de texto (pymupdf), com a
posição na página, o tamanho da fonte, a fração em negrito e a fração de linhas
com texto girado (tabelas em paisagem). NÃO limpa nada: a
limpeza é o passo 02. Guardar a posição e a fonte permite, depois, detectar
cabeçalhos/rodapés, títulos e legendas sem depender só do texto.

Entrada : data/raw/*.pdf
Saída   : data/processed/blocos_raw.parquet

Uso (da raiz do projeto):
    uv run python scripts/01_extrair_blocos.py
    uv run python scripts/01_extrair_blocos.py --raw data/raw --saida data/processed/blocos_raw.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pymupdf


def extrair_pdf(caminho: Path):
    """Devolve (linhas, n_paginas, paginas_sem_texto) de um PDF."""
    linhas = []
    paginas_sem_texto = []
    with pymupdf.open(caminho) as pdf:
        n_paginas = pdf.page_count
        for pno, page in enumerate(pdf, start=1):
            larg, alt = page.rect.width, page.rect.height
            # sort=True ordena os blocos de cima para baixo e da esquerda para a direita
            d = page.get_text("dict", sort=True)
            n_blocos_pagina = 0
            for b in d["blocks"]:
                if b.get("type") != 0:  # 0 = texto; 1 = imagem
                    continue
                textos, soma_tam, soma_neg, soma_car = [], 0.0, 0, 0
                n_girado = 0  # linhas que não correm da esquerda para a direita
                for ln in b["lines"]:
                    spans = ln["spans"]
                    t = "".join(s["text"] for s in spans)
                    if not t.strip():
                        continue
                    textos.append(t)
                    if ln["dir"][0] < 0.9:
                        n_girado += 1
                    for s in spans:
                        k = len(s["text"].strip())
                        soma_car += k
                        soma_tam += s["size"] * k
                        if (s["flags"] & 16) or ("bold" in s["font"].lower()):
                            soma_neg += k
                if not textos or soma_car == 0:
                    continue
                x0, y0, x1, y1 = b["bbox"]
                n_blocos_pagina += 1
                linhas.append(
                    {
                        "doc": caminho.stem,
                        "pagina": pno,
                        "bloco": n_blocos_pagina,
                        "x0": round(x0, 1),
                        "y0": round(y0, 1),
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "larg_pag": round(larg, 1),
                        "alt_pag": round(alt, 1),
                        "tam_fonte": round(soma_tam / soma_car, 2),
                        "negrito": round(soma_neg / soma_car, 2),
                        "n_linhas": len(textos),
                        "girado": round(n_girado / len(textos), 2),
                        "texto": "\n".join(textos),
                    }
                )
            if n_blocos_pagina == 0:
                paginas_sem_texto.append(pno)
    return linhas, n_paginas, paginas_sem_texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="pasta com os PDFs (padrão: data/raw)")
    ap.add_argument("--saida", default="data/processed/blocos_raw.parquet")
    args = ap.parse_args()

    raw = Path(args.raw)
    pdfs = sorted(raw.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF encontrado em {raw}/. Rode o passo de download antes.")

    todas = []
    print(f"{'documento':<28}{'páginas':>8}{'blocos':>8}{'sem texto':>11}")
    for pdf in pdfs:
        linhas, n_pag, sem_texto = extrair_pdf(pdf)
        todas.extend(linhas)
        print(f"{pdf.stem:<28}{n_pag:>8}{len(linhas):>8}{len(sem_texto):>11}")
        if n_pag and len(sem_texto) / n_pag > 0.05:
            print(
                f"  AVISO: {len(sem_texto)} de {n_pag} páginas sem texto extraível "
                f"(ex.: {sem_texto[:5]}). Pode ser PDF escaneado; nesse caso seria preciso OCR."
            )

    df = pd.DataFrame(todas)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(saida, index=False)
    print(f"\n{len(df)} blocos gravados em {saida}")


if __name__ == "__main__":
    main()
