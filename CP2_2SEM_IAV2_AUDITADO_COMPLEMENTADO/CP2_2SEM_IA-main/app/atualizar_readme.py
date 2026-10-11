"""
Preenche as seções "Resultados da execução" e "Análise dos resultados" do README.md
com os números REAIS gerados por app/avaliar_rag.py (resultados/vencedora.json
e resultados/ragas_por_pergunta.csv). Nada é inventado: se a avaliação ainda não
foi executada, o script avisa e não altera o README.

Uso:
    python -m app.atualizar_readme
"""
import csv
import os
import sys

PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_PROJETO not in sys.path:
    sys.path.insert(0, PASTA_PROJETO)

from app.config_rag import carregar_vencedora, PASTA_RESULTADOS

INICIO = "<!-- RESULTADOS_RAGAS_INICIO -->"
FIM = "<!-- RESULTADOS_RAGAS_FIM -->"
META_APROVACAO = 0.7
META_IDEAL = 0.9


def _faixa(faithfulness: float) -> str:
    if faithfulness >= META_IDEAL:
        return "zona ideal (≥ 0,9)"
    if faithfulness >= META_APROVACAO:
        return "aprovada (≥ 0,7)"
    return "ABAIXO da meta de 0,7"


def _ler_csv() -> list:
    caminho = os.path.join(PASTA_RESULTADOS, "ragas_por_pergunta.csv")
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _numero(texto: str):
    try:
        valor = float(texto)
        return None if valor != valor else valor  # descarta NaN
    except (TypeError, ValueError):
        return None


def gerar_bloco(dados: dict, linhas_csv: list) -> str:
    """Monta o Markdown com a tabela de resumo, a vencedora e a pior pergunta de cada configuração."""
    resumos = sorted(dados["resumos"], key=lambda r: r["chunk_size"])
    venc = dados["vencedora"]
    out = ["| chunk_size | Chunks gerados | Faithfulness | Answer relevancy |",
           "|-----------:|---------------:|-------------:|-----------------:|"]
    for r in resumos:
        marca = " ✅" if r["chunk_size"] == venc["chunk_size"] else ""
        out.append(f"| {r['chunk_size']}{marca} | {r['chunks']} | {r['faithfulness']:.3f} | "
                   f"{r['answer_relevancy']:.3f} |")
    out.append("")
    out.append(f"**Configuração vencedora: `chunk_size = {venc['chunk_size']}`** — faithfulness médio "
               f"{venc['faithfulness']:.3f} ({_faixa(venc['faithfulness'])}) e answer relevancy médio "
               f"{venc['answer_relevancy']:.3f}. Critério: maior faithfulness, com answer relevancy como desempate.")

    if len(resumos) > 1:
        outros = [r for r in resumos if r["chunk_size"] != venc["chunk_size"]]
        comp = "; ".join(f"`{r['chunk_size']}` ficou com {r['faithfulness']:.3f} "
                         f"({r['faithfulness'] - venc['faithfulness']:+.3f} em relação à vencedora)"
                         for r in outros)
        out.append("")
        out.append(f"Nas demais configurações: {comp}.")

    # Pior pergunta (menor faithfulness) da configuração vencedora
    da_vencedora = [l for l in linhas_csv if str(l.get("chunk_size")) == str(venc["chunk_size"])]
    validas = [(l, _numero(l.get("faithfulness"))) for l in da_vencedora]
    validas = [(l, v) for l, v in validas if v is not None]
    if validas:
        pior, nota = min(validas, key=lambda par: par[1])
        out.append("")
        out.append(f"Pergunta com menor faithfulness na vencedora ({nota:.3f}): \"{pior['pergunta']}\". "
                   "Veja a análise de falhas no notebook (seção 12) para o diagnóstico dos chunks recuperados.")

    out.append("")
    out.append("> A pergunta fora do domínio (geladeira) é recusada de propósito e recebe answer relevancy 0 "
               "no RAGAS, o que reduz a média dessa métrica.")
    out.append("")
    out.append("_Tabela por pergunta: `resultados/ragas_por_pergunta.csv`. Gerado por `python -m app.atualizar_readme`._")
    return "\n".join(out)


def atualizar(caminho_readme: str | None = None) -> bool:
    caminho_readme = caminho_readme or os.path.join(PASTA_PROJETO, "README.md")
    dados = carregar_vencedora()
    if not dados:
        print("resultados/vencedora.json não encontrado: rode antes `python -m app.avaliar_rag` "
              "(ou a seção 11 do notebook). README não alterado.")
        return False
    with open(caminho_readme, encoding="utf-8") as f:
        texto = f.read()
    if INICIO not in texto or FIM not in texto:
        print(f"Marcadores {INICIO} / {FIM} não encontrados no README.")
        return False
    antes, resto = texto.split(INICIO, 1)
    _, depois = resto.split(FIM, 1)
    novo = f"{antes}{INICIO}\n{gerar_bloco(dados, _ler_csv())}\n{FIM}{depois}"
    with open(caminho_readme, "w", encoding="utf-8") as f:
        f.write(novo)
    print("README.md atualizado com os resultados reais do RAGAS.")
    return True


if __name__ == "__main__":
    atualizar()
