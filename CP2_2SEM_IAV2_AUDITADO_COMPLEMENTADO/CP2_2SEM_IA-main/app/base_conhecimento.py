"""
Base de conhecimento do CKP02 (etapa LOAD do pipeline RAG).

Carrega os documentos da pasta docs/ (TXT, Markdown ou PDF) e anexa metadados
(produto, categoria, tipo, fonte, data) a partir de docs/catalogo.json.
Esses metadados permitem o metadata filtering com where= (Aula 05, Slide 12).
"""
import json
import os
import re

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader

NOME_CATALOGO = "catalogo.json"
EXTENSOES_TEXTO = (".txt", ".md")
EXTENSOES_VALIDAS = EXTENSOES_TEXTO + (".pdf",)


def carregar_catalogo(docs_dir: str) -> dict:
    """Lê docs/catalogo.json e devolve {nome_do_arquivo: metadados}."""
    caminho = os.path.join(docs_dir, NOME_CATALOGO)
    if not os.path.exists(caminho):
        return {}
    with open(caminho, encoding="utf-8") as f:
        itens = json.load(f)
    return {item["arquivo"]: item for item in itens}


def _produto_do_cabecalho(texto: str) -> str:
    """Se o arquivo não estiver no catálogo, tenta ler a linha 'Produto: ...'."""
    achado = re.search(r"^Produto:\s*(.+?)\.?\s*$", texto, flags=re.MULTILINE)
    return achado.group(1) if achado else ""


def _metadados_validos(meta: dict) -> dict:
    """ChromaDB só aceita str, int, float e bool nos metadados (sem None)."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def carregar_documentos(docs_dir: str) -> list:
    """
    Carrega todos os documentos da pasta e devolve uma lista de Document.
    TXT/MD viram 1 Document por arquivo; PDF vira 1 Document por página (Aula 06).
    """
    catalogo = carregar_catalogo(docs_dir)
    documentos = []

    # Avisa se algum documento do catálogo ainda não foi colocado na pasta
    for nome in catalogo:
        if not os.path.exists(os.path.join(docs_dir, nome)):
            print(f"AVISO: '{nome}' está em {NOME_CATALOGO} mas não está na pasta {docs_dir}/.")

    for nome in sorted(os.listdir(docs_dir)):
        caminho = os.path.join(docs_dir, nome)
        extensao = os.path.splitext(nome)[1].lower()
        if extensao not in EXTENSOES_VALIDAS:
            continue

        if extensao == ".pdf":
            paginas = PyMuPDFLoader(caminho).load()
        else:
            paginas = TextLoader(caminho, encoding="utf-8").load()

        info = catalogo.get(nome, {})
        for pagina in paginas:
            pagina.metadata.update(_metadados_validos({
                "arquivo": nome,
                "produto": info.get("produto") or _produto_do_cabecalho(pagina.page_content) or nome,
                "marca": info.get("marca", ""),
                "categoria": info.get("categoria", "geral"),
                "tipo": info.get("tipo", "manual"),
                "formato_original": info.get("formato_original", extensao.lstrip(".")),
                "fonte_url": info.get("fonte_url", ""),
                "data_inclusao": info.get("data_inclusao", ""),
            }))
        documentos.extend(paginas)

    return documentos


def listar_categorias(documentos: list) -> list:
    """Categorias distintas presentes na base (usadas nos filtros da interface)."""
    return sorted({d.metadata.get("categoria", "geral") for d in documentos})
