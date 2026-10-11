import os

from pydantic import ValidationError

from .chains import chat_chain, analisar_atendimento
from .memory_manager import visualizar_memoria, limpar_memoria
from .rag_service import SuporteRAGService, formatar_fontes
from .config_rag import chunk_size_vencedor

PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iniciar_chat():
    print("==================================================")
    print(" Suporte Técnico de Aparelhos Eletrônicos (CKP02)")
    print("==================================================")
    print("Comandos disponíveis:")
    print("  'rag: <sua dúvida>' -> Consulta direta aos Manuais Oficiais via RAG")
    print("  'filtro <categoria>' -> Restringe o RAG a uma categoria (ex.: filtro notebook; filtro todas)")
    print("  'rerank'            -> Liga/desliga o reranking com cross-encoder")
    print("  'memoria'           -> Visualiza histórico da conversa")
    print("  'limpar'            -> Limpa histórico da sessão")
    print("  'sair'              -> Encerra o programa\n")

    # Inicializa o serviço RAG modular com os manuais de suporte
    # chunk_size = configuração vencedora da avaliação RAGAS (resultados/vencedora.json);
    # se a avaliação ainda não foi rodada, usa o padrão de app/config_rag.py
    servico_rag = SuporteRAGService(
        docs_dir=os.path.join(PASTA_PROJETO, "docs"),
        chunk_size=chunk_size_vencedor(),
        persist_dir=os.path.join(PASTA_PROJETO, "chroma_db"),
    )
    categoria_ativa = "todas"
    reranking_ativo = False
    print(f"Categorias da base: {', '.join(servico_rag.categorias())}\n")

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            break

        if pergunta.lower() == "limpar":
            limpar_memoria(chat_chain.memory)
            print("Memória limpa com sucesso.\n")
            continue

        partes = pergunta.lower().split()
        if partes and partes[0] == "filtro" and len(partes) <= 2:
            categoria_ativa = partes[1] if len(partes) == 2 else "todas"
            print(f"Filtro do RAG: {categoria_ativa}\n")
            continue

        if pergunta.lower() == "rerank":
            reranking_ativo = not reranking_ativo
            print(f"Reranking: {'ligado' if reranking_ativo else 'desligado'}\n")
            continue

        if pergunta.lower() == "memoria":
            estado = visualizar_memoria(chat_chain.memory)
            print(estado)
            print()
            continue

        if not pergunta:
            print("Por favor, digite uma mensagem válida.\n")
            continue

        print("Processando...")

        # Se o usuário digitar 'rag:', direcionamos para o pipeline RAG do CKP02
        if pergunta.lower().startswith("rag:"):
            consulta_rag = pergunta[4:].strip()
            resultado_rag = servico_rag.consultar(
                consulta_rag, categoria=categoria_ativa, usar_reranking=reranking_ativo
            )
            print(f"\n[Base de Manuais - RAG]:\n{resultado_rag['resposta']}\n")
            print(f"{formatar_fontes(resultado_rag['fontes'])}\n")
            continue

        # Fluxo normal conversacional do CKP01 com memória e saída estruturada
        resposta = chat_chain.predict(input=pergunta)
        print(f"Assistente: {resposta}\n")

        try:
            analise = analisar_atendimento(pergunta, resposta)
            print("Análise estruturada do atendimento:")
            print(analise.model_dump())
            print()
        except ValidationError as erro:
            print(f"Aviso de validação Pydantic: {erro}\n")
        except Exception as erro:
            print(f"Falha na análise estruturada: {erro}\n")


if __name__ == "__main__":
    iniciar_chat()