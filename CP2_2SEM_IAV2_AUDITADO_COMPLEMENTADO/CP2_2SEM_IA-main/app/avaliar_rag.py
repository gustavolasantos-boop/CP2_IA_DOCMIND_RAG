"""
Avaliação do CKP02: compara configurações de chunk_size com RAGAS
(faithfulness + answer_relevancy), por pergunta e na média.

Usa o MESMO pipeline do chatbot (app/rag_service.py), então o que é medido
é exatamente o que o usuário recebe. Lê os documentos da pasta docs/.

Exemplos:
    python -m app.avaliar_rag                      # compara 256, 512 e 1024 (como a demo da Aula 07)
    python -m app.avaliar_rag --chunks 512 1024
    python -m app.avaliar_rag --rerank             # mede com reranking ligado
"""
import argparse
import os
import sys

# Permite rodar também como "python app/avaliar_rag.py" (PyCharm)
PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_PROJETO not in sys.path:
    sys.path.insert(0, PASTA_PROJETO)

from dotenv import load_dotenv

load_dotenv(os.path.join(PASTA_PROJETO, ".env"))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.rag_service import SuporteRAGService
from app.perguntas_teste import PERGUNTAS as _PERGUNTAS
from app.config_rag import salvar_vencedora

PASTA_DOCS = os.path.join(PASTA_PROJETO, "docs")
PASTA_RESULTADOS = os.path.join(PASTA_PROJETO, "resultados")

# Perguntas de teste (com documento, página e resposta esperada) em app/perguntas_teste.py
PERGUNTAS = _PERGUNTAS

FAITHFULNESS_APROVACAO = 0.7  # meta do CKP02 (aprovação)
FAITHFULNESS_IDEAL = 0.9      # zona ideal
CHUNKS_PADRAO = [256, 512, 1024]  # mesmas configurações da demo da Aula 07


def avaliar_configuracao(chunk_size: int, perguntas: list = PERGUNTAS, usar_reranking: bool = False,
                         docs_dir: str = PASTA_DOCS):
    """
    Cria o pipeline com o chunk_size dado, responde as perguntas e mede com RAGAS.
    Devolve (tabela_por_pergunta, resumo_da_configuracao).
    """
    print(f"\n=======================================================")
    print(f" TESTANDO CONFIGURAÇÃO: chunk_size = {chunk_size}"
          f"{' + reranking' if usar_reranking else ''}")
    print(f"=======================================================")

    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Pasta docs não encontrada: {docs_dir}")

    # Mesmo serviço do chatbot, com ChromaDB em memória (chromadb.Client)
    # Coleção própria da avaliação: apagá-la no fim não afeta o chatbot nem a interface
    servico = SuporteRAGService(docs_dir=docs_dir, chunk_size=chunk_size, usar_reranking=usar_reranking,
                                nome_colecao=f"avaliacao_{chunk_size}")
    if servico.chain is None:
        raise ValueError("Nenhum documento (.txt, .md ou .pdf) encontrado na pasta docs.")
    print(f"Total de chunks gerados: {len(servico.chunks)}")

    # RAGAS usa o mesmo LLM (gemma4:cloud, temperature=0) e os mesmos embeddings do pipeline
    ragas_llm = LangchainLLMWrapper(servico.llm)
    ragas_embs = LangchainEmbeddingsWrapper(servico.embeddings)
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embs

    dados = {"question": [], "answer": [], "contexts": []}
    for i, p in enumerate(perguntas, 1):
        print(f"\n--- Pergunta {i} ---")
        print(f"Pergunta: {p}")
        resultado = servico.consultar(p)
        print(f"Resposta:\n{resultado['resposta']}")
        print("Fontes: " + "; ".join(f"{f['arquivo']} (trecho {f['trecho']})" for f in resultado["fontes"]))

        dados["question"].append(p)
        dados["answer"].append(resultado["resposta"])
        dados["contexts"].append([d.page_content for d in resultado["documentos"]])

    # Avaliação com RAGAS
    resultado_ragas = evaluate(
        Dataset.from_dict(dados),
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embs,
        raise_exceptions=True
    )
    tabela = resultado_ragas.to_pandas()
    coluna_pergunta = "user_input" if "user_input" in tabela.columns else "question"
    tabela = tabela[[coluna_pergunta, "faithfulness", "answer_relevancy"]].rename(
        columns={coluna_pergunta: "pergunta"})
    tabela.insert(0, "chunk_size", chunk_size)
    print("\nRESULTADOS POR PERGUNTA:")
    print(tabela.drop(columns="chunk_size").to_string(index=False))

    resumo = {
        "chunk_size": chunk_size,
        "chunks": len(servico.chunks),
        "faithfulness": tabela["faithfulness"].mean(),
        "answer_relevancy": tabela["answer_relevancy"].mean(),
    }
    servico.descartar_colecao()
    return tabela, resumo


def escolher_vencedora(resumos: list) -> dict:
    """Maior faithfulness vence; answer_relevancy desempata (critério do CKP02)."""
    return max(resumos, key=lambda r: (round(r["faithfulness"], 3), r["answer_relevancy"]))


def salvar_resultados(tabelas: list, resumos: list, vencedora: dict):
    """Grava CSV por pergunta e um resumo em Markdown na pasta resultados/."""
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    pd.concat(tabelas).to_csv(os.path.join(PASTA_RESULTADOS, "ragas_por_pergunta.csv"),
                              index=False, encoding="utf-8")

    linhas = ["| chunk_size | Chunks gerados | Faithfulness | Answer relevancy |",
              "|-----------:|---------------:|-------------:|-----------------:|"]
    for r in resumos:
        linhas.append(f"| {r['chunk_size']} | {r['chunks']} | "
                      f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} |")
    linhas.append("")
    linhas.append(f"Configuração vencedora: chunk_size = {vencedora['chunk_size']} "
                  f"(faithfulness {vencedora['faithfulness']:.3f}, "
                  f"answer_relevancy {vencedora['answer_relevancy']:.3f}).")
    with open(os.path.join(PASTA_RESULTADOS, "ragas_resumo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    # Registra a vencedora para o chatbot, a interface Gradio e o CKP03 usarem a mesma configuração
    salvar_vencedora(vencedora, resumos)


def diagnosticar_pior_pergunta(tabelas: list, servico, chunk_size: int | None = None) -> dict:
    """
    Análise de falhas (Aula 07, Exercício 4): acha a pergunta com menor faithfulness
    e mostra os chunks que o retriever entregou ao LLM para ela.

    Regra de leitura: trecho PRESENTE no contexto + resposta ruim -> problema no prompt/LLM;
    trecho AUSENTE do contexto -> problema no retriever (chunk_size/overlap/k).
    """
    tabela = pd.concat(tabelas)
    if chunk_size is not None:
        tabela = tabela[tabela["chunk_size"] == chunk_size]
    linha = tabela.sort_values("faithfulness").iloc[0]
    pergunta = linha["pergunta"]
    print(f"Pergunta crítica (faithfulness {linha['faithfulness']:.3f}): {pergunta}\n")
    encontrados = servico.recuperar(pergunta)
    for i, (doc, sim, _) in enumerate(encontrados, 1):
        print(f"[chunk {i}] {doc.metadata.get('arquivo')} (trecho {doc.metadata.get('trecho')}, "
              f"similaridade {sim:.3f})\n{doc.page_content[:300]}\n")
    return {"pergunta": pergunta, "faithfulness": float(linha["faithfulness"]), "chunks": encontrados}


def comparar(chunk_sizes: list, usar_reranking: bool = False, perguntas: list = PERGUNTAS):
    """Roda todas as configurações e devolve (tabelas, resumos, vencedora)."""
    tabelas, resumos = [], []
    for tamanho in chunk_sizes:
        tabela, resumo = avaliar_configuracao(tamanho, perguntas, usar_reranking)
        tabelas.append(tabela)
        resumos.append(resumo)
    return tabelas, resumos, escolher_vencedora(resumos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara configurações de chunking com RAGAS.")
    parser.add_argument("--chunks", type=int, nargs="+", default=CHUNKS_PADRAO,
                        help="valores de chunk_size a comparar (padrão: 256 512 1024)")
    parser.add_argument("--rerank", action="store_true", help="liga o reranking com cross-encoder")
    args = parser.parse_args()

    tabelas, resumos, vencedora = comparar(args.chunks, args.rerank)
    salvar_resultados(tabelas, resumos, vencedora)

    print("\n=======================================================")
    print(" COMPARAÇÃO DAS MÉDIAS FINAIS")
    print("=======================================================")
    print("chunk_size | chunks | faithfulness | answer_relevancy")
    for r in resumos:
        print(f"{r['chunk_size']:10d} | {r['chunks']:6d} | {r['faithfulness']:.3f}        | {r['answer_relevancy']:.3f}")
    print(f"\nVencedora: chunk_size = {vencedora['chunk_size']}")
    status = ("zona ideal" if vencedora["faithfulness"] >= FAITHFULNESS_IDEAL
              else "aprovada" if vencedora["faithfulness"] >= FAITHFULNESS_APROVACAO
              else "ABAIXO DA META (0,7): itere no chunking")
    print(f"Faithfulness da vencedora: {vencedora['faithfulness']:.3f} → {status}")
    print(f"Resultados salvos em: {PASTA_RESULTADOS}")
