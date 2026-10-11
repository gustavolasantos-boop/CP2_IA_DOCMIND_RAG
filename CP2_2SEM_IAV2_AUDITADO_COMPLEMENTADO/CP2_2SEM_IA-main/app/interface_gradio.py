"""
Interface Gradio simples para o RAG do CKP02 (diferencial).

Executar:
    python -m app.interface_gradio
Para gerar um link público (ex.: Colab), defina GRADIO_SHARE=1.
"""
import os
import sys

PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_PROJETO not in sys.path:
    sys.path.insert(0, PASTA_PROJETO)

import gradio as gr

from app.rag_service import SuporteRAGService, formatar_fontes
from app.config_rag import chunk_size_vencedor


def criar_interface(servico: SuporteRAGService):
    """Monta o gr.ChatInterface com filtro por categoria, reranking e fonte citada."""

    def responder(mensagem, historico, categoria, usar_reranking):
        try:
            resultado = servico.consultar(mensagem, categoria=categoria, usar_reranking=usar_reranking)
        except Exception as erro:  # ex.: sem OLLAMA_API_KEY ou sem sentence-transformers
            return f"Não foi possível consultar a base: {erro}"
        fontes = formatar_fontes(resultado["fontes"])
        return resultado["resposta"] + (f"\n\n{fontes}" if fontes else "")

    return gr.ChatInterface(
        fn=responder,
        title="DocMind RAG · Suporte Técnico de Aparelhos Eletrônicos",
        description=("Pergunte sobre os manuais da base. A resposta cita o arquivo e o trecho de origem. "
                     "Use o filtro para restringir a busca a uma categoria (metadata filtering)."),
        additional_inputs=[
            gr.Dropdown(["todas"] + servico.categorias(), value="todas", label="Categoria (filtro)"),
            gr.Checkbox(value=False, label="Reranking (cross-encoder)"),
        ],
        examples=[
            ["Como restaurar o roteador Intelbras para as configurações de fábrica?", "roteador", False],
            ["O que fazer quando a TV está sem imagem?", "tv", True],
        ],
    )


if __name__ == "__main__":
    servico = SuporteRAGService(
        docs_dir=os.path.join(PASTA_PROJETO, "docs"),
        chunk_size=chunk_size_vencedor(),  # vencedora do RAGAS (ou RAG_CHUNK_SIZE)
        persist_dir=os.path.join(PASTA_PROJETO, "chroma_db"),
    )
    criar_interface(servico).launch(share=os.getenv("GRADIO_SHARE") == "1")
