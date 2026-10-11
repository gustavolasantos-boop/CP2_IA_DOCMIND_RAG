"""
Checklist automático do CKP02: confere, item a item, os requisitos do PDF do professor
antes de enviar no Teams. Não chama nenhuma API (roda offline).

Uso:
    python -m app.verificar_entrega
"""
import csv
import glob
import json
import os
import re
import sys

PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_PROJETO not in sys.path:
    sys.path.insert(0, PASTA_PROJETO)

from app.config_rag import carregar_vencedora, PASTA_RESULTADOS

CAMPOS_CATALOGO = ["arquivo", "produto", "categoria", "tipo", "fonte_url", "data_inclusao"]


def _ler(caminho: str) -> str:
    with open(caminho, encoding="utf-8") as f:
        return f.read()


def verificar() -> list:
    """Devolve lista de (ok, requisito, detalhe)."""
    r = []
    add = lambda ok, req, det="": r.append((ok, req, det))

    # --- Base de conhecimento (2,0 pts) ---
    docs = os.path.join(PASTA_PROJETO, "docs")
    arquivos = [a for a in os.listdir(docs) if a.lower().endswith((".pdf", ".txt", ".md"))] if os.path.isdir(docs) else []
    add(len(arquivos) >= 5, "≥ 5 documentos em docs/", f"{len(arquivos)} encontrados")
    try:
        catalogo = json.loads(_ler(os.path.join(docs, "catalogo.json")))
    except Exception as e:
        catalogo = []
        add(False, "docs/catalogo.json legível", str(e))
    sem_fonte = [c["arquivo"] for c in catalogo if not str(c.get("fonte_url", "")).startswith("http")]
    add(bool(catalogo) and not sem_fonte, "Todos os documentos com fonte_url (link de origem)", ", ".join(sem_fonte))
    faltando = [c["arquivo"] for c in catalogo for k in CAMPOS_CATALOGO if not c.get(k)]
    add(not faltando, "Metadados completos (arquivo, produto, categoria, tipo, fonte, data)", ", ".join(sorted(set(faltando))))
    nao_listados = [a for a in arquivos if a not in {c["arquivo"] for c in catalogo}]
    add(not nao_listados, "Todo arquivo de docs/ está no catálogo", ", ".join(nao_listados))
    pequenos = [a for a in arquivos if os.path.getsize(os.path.join(docs, a)) < 20_000]
    add(not pequenos, "Documentos são os originais, não resumos curtos (≥ 20 KB cada)",
        "Atenção, arquivo pequeno (resumo?): " + ", ".join(pequenos) if pequenos else "")

    # --- Pipeline / código (3,5 + 1,5 pts) ---
    svc = _ler(os.path.join(PASTA_PROJETO, "app", "rag_service.py"))
    add('model="nomic-embed-text"' in svc, "Embeddings: OllamaEmbeddings(model='nomic-embed-text')")
    add('model="gemma4:cloud", temperature=0' in svc, "LLM: gemma4:cloud com temperature=0")
    add('separators=["\\n\\n", "\\n", ". ", " ", ""]' in svc, "RecursiveCharacterTextSplitter com os separators do professor")
    add("chunk_size // 8" in svc, "chunk_overlap = chunk_size // 8 (12,5%, dentro de 10–15%)")
    add("PersistentClient" in svc, "ChromaDB local (PersistentClient/Client)")
    add(re.search(r"^def buscar\(", svc, flags=re.M) is not None, "Função modular buscar(consulta) para o CKP03")
    add("where" in _ler(os.path.join(PASTA_PROJETO, "app", "base_conhecimento.py")) or "filter=filtro" in svc,
        "Diferencial: metadata filtering")
    add("ms-marco-MiniLM-L-6-v2" in svc, "Diferencial: reranking com cross-encoder")
    add(os.path.exists(os.path.join(PASTA_PROJETO, "app", "interface_gradio.py")), "Diferencial: interface Gradio")

    # --- RAGAS (3,0 pts) ---
    dados = carregar_vencedora()
    add(dados is not None, "Avaliação RAGAS executada (resultados/vencedora.json)",
        "rode: python -m app.avaliar_rag" if dados is None else "")
    if dados:
        add(len(dados["resumos"]) >= 2, "≥ 2 configurações de chunk_size comparadas", f"{len(dados['resumos'])} configs")
        venc = dados["vencedora"]
        add(venc["faithfulness"] >= 0.7, "Faithfulness da vencedora ≥ 0,7", f"{venc['faithfulness']:.3f}")
        caminho_csv = os.path.join(PASTA_RESULTADOS, "ragas_por_pergunta.csv")
        if os.path.exists(caminho_csv):
            with open(caminho_csv, encoding="utf-8", newline="") as f:
                linhas = list(csv.DictReader(f))
            por_config = {}
            for l in linhas:
                por_config.setdefault(l["chunk_size"], set()).add(l["pergunta"])
            minimo = min((len(v) for v in por_config.values()), default=0)
            add(minimo >= 5, "≥ 5 perguntas testadas por configuração", f"mínimo {minimo}")
        else:
            add(False, "resultados/ragas_por_pergunta.csv existe")

    # --- README ---
    readme = _ler(os.path.join(PASTA_PROJETO, "README.md"))
    add(len(re.findall(r"RM\s*\d{5,6}", readme)) >= 3, "README com nomes e RMs dos integrantes",
        f"{len(re.findall(r'RM' + chr(92) + 's*' + chr(92) + 'd{5,6}', readme))} RMs")
    add("Como adicionar novos documentos" in readme, "README com instruções para adicionar novos documentos")
    ini, fim = "<!-- RESULTADOS_RAGAS_INICIO -->", "<!-- RESULTADOS_RAGAS_FIM -->"
    bloco = readme.split(ini, 1)[1].split(fim, 1)[0] if ini in readme and fim in readme else ""
    add("Configuração vencedora" in bloco, "README com resultados RAGAS preenchidos",
        "rode: python -m app.atualizar_readme" if "Configuração vencedora" not in bloco else "")

    # --- Notebook (Run All sem erros) ---
    nbs = glob.glob(os.path.join(PASTA_PROJETO, "CKP02_*_grupo*.ipynb"))
    add(len(nbs) == 1, "Notebook nomeado CKP02_[dominio]_grupo.ipynb", ", ".join(map(os.path.basename, nbs)))
    if nbs:
        nb = json.loads(_ler(nbs[0]))
        codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
        executadas = [c for c in codigo if c.get("outputs")]
        erros = [c for c in codigo for o in c.get("outputs", []) if o.get("output_type") == "error"]
        add(len(executadas) >= len(codigo) - 3, "Notebook exportado JÁ EXECUTADO (com saídas)",
            f"{len(executadas)}/{len(codigo)} células com saída")
        add(not erros, "Notebook sem células com erro", f"{len(erros)} com erro")

    # --- Segurança ---
    add(not os.path.exists(os.path.join(PASTA_PROJETO, ".env")), "Sem .env no pacote de entrega (chave secreta)")
    return r


if __name__ == "__main__":
    resultado = verificar()
    for ok, req, det in resultado:
        print(f"[{'OK' if ok else '  '}] {req}" + (f"  → {det}" if det else ""))
    pendentes = [x for x in resultado if not x[0]]
    print(f"\n{len(resultado) - len(pendentes)}/{len(resultado)} itens OK.")
    sys.exit(1 if pendentes else 0)
