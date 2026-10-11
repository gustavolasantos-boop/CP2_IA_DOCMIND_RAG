"""
Configuração compartilhada do RAG (CKP02).

Guarda qual chunk_size venceu na avaliação RAGAS para que o chatbot (main.py),
a interface Gradio e o notebook usem sempre a MESMA configuração medida.
O arquivo resultados/vencedora.json é gerado por app/avaliar_rag.py.
"""
import json
import os

PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RESULTADOS = os.path.join(PASTA_PROJETO, "resultados")
ARQUIVO_VENCEDORA = os.path.join(PASTA_RESULTADOS, "vencedora.json")

# Usado apenas enquanto a avaliação RAGAS ainda não foi executada.
CHUNK_SIZE_PADRAO = 1024


def _normalizar(item: dict) -> dict:
    """chunk_size e chunks viram int; as métricas viram float."""
    return {k: (int(v) if k in ("chunk_size", "chunks") else float(v)) for k, v in item.items()}


def salvar_vencedora(vencedora: dict, resumos: list) -> None:
    """Grava a configuração vencedora e o resumo de todas as configurações testadas."""
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    conteudo = {"vencedora": _normalizar(vencedora), "resumos": [_normalizar(r) for r in resumos]}
    with open(ARQUIVO_VENCEDORA, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2)


def carregar_vencedora() -> dict | None:
    """Lê resultados/vencedora.json; devolve None se a avaliação ainda não foi rodada."""
    if not os.path.exists(ARQUIVO_VENCEDORA):
        return None
    with open(ARQUIVO_VENCEDORA, encoding="utf-8") as f:
        return json.load(f)


def chunk_size_vencedor() -> int:
    """
    chunk_size a usar no chatbot e na interface.
    Ordem: variável RAG_CHUNK_SIZE > vencedora medida pelo RAGAS > padrão.
    """
    manual = os.getenv("RAG_CHUNK_SIZE")
    if manual:
        return int(manual)
    dados = carregar_vencedora()
    if dados:
        return int(dados["vencedora"]["chunk_size"])
    return CHUNK_SIZE_PADRAO
