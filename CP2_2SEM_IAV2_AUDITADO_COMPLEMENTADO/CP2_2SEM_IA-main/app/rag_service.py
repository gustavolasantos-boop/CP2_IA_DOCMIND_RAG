"""
Pipeline RAG do CKP02: load -> split -> embed -> store -> retrieve -> generate.

Base: Aulas 05 (embeddings + ChromaDB), 06 (pipeline RAG) e 07 (reranking).
Reutilizável como @tool no CKP03 pela função SuporteRAGService.buscar(consulta).
"""
import hashlib
import os
from dotenv import load_dotenv

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .base_conhecimento import carregar_documentos, listar_categorias
from .config_rag import PASTA_PROJETO, chunk_size_vencedor

load_dotenv()

NOME_DOMINIO = "suporte_eletronicos"
MENSAGEM_SEM_RESPOSTA = "Não encontrei essa informação nos documentos fornecidos."
MODELO_RERANKING = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Aula 07, Slide 09

# Prompt estrito de grounding conforme a Aula 06 / Slide 12.
# A citação agora usa o rótulo de cada trecho recuperado (arquivo + número do trecho).
PROMPT_RAG_TEXTO = """<persona>
Você é um assistente técnico especializado em suporte de aparelhos eletrônicos,
respondendo com base EXCLUSIVAMENTE nos manuais e documentos fornecidos.
</persona>

<instrucoes>
- Responda SOMENTE com informações do contexto abaixo.
- Sempre cite a fonte ao final, usando o rótulo do trecho usado:
  (fonte: nome-do-arquivo, trecho N, pág. P). Omita "pág." se o rótulo não tiver página.
- Se a resposta não estiver no contexto, responda exatamente:
  "Não encontrei essa informação nos documentos fornecidos."
- Nunca invente, extrapole ou use conhecimento prévio externo.
</instrucoes>

<contexto>
{contexto}
</contexto>

<pergunta>
{pergunta}
</pergunta>"""


def configurar_ollama_cloud():
    """Configura a Ollama Cloud (OLLAMA_HOST + OLLAMA_API_KEY) como nas aulas."""
    os.environ["OLLAMA_HOST"] = "https://ollama.com"
    chave = os.getenv("OLLAMA_API_KEY")
    if chave:
        os.environ["OLLAMA_API_KEY"] = chave


def montar_filtro(**campos) -> dict | None:
    """
    Monta o where= do ChromaDB (Aula 05, Slide 12).
    Um campo -> {"categoria": "notebook"}; vários -> {"$and": [...]}.
    Valores vazios ou "todas" são ignorados (sem filtro).
    """
    condicoes = [{k: v} for k, v in campos.items() if v and v != "todas"]
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


_cross_encoder_cache = {}


def _obter_cross_encoder():
    """Carrega o cross-encoder uma única vez (roda em CPU, Aula 07)."""
    if "modelo" not in _cross_encoder_cache:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        _cross_encoder_cache["modelo"] = HuggingFaceCrossEncoder(model_name=MODELO_RERANKING)
    return _cross_encoder_cache["modelo"]


class SuporteRAGService:
    def __init__(
        self,
        docs_dir: str = "docs",
        chunk_size: int = 512,
        k: int = 3,
        usar_reranking: bool = False,
        k_busca_ampla: int = 10,
        persist_dir: str | None = None,
        nome_colecao: str | None = None,
        embeddings=None,
        llm=None,
        reranker=None,
    ):
        configurar_ollama_cloud()

        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_size // 8  # 12,5% do chunk_size (faixa de 10-15% do CKP02)
        self.k = k
        self.usar_reranking = usar_reranking
        self.k_busca_ampla = k_busca_ampla  # candidatos que o reranker recebe (Aula 07)
        self.persist_dir = persist_dir
        self.nome_colecao = nome_colecao or f"{NOME_DOMINIO}_{chunk_size}"
        self.reranker = reranker  # opcional: objeto com .score([(consulta, texto), ...])

        # Modelo e provedor de embeddings fixados pelo professor:
        # nomic-embed-text via Ollama Cloud (configurada acima).
        self.embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")

        # LLM fixado pelo professor: gemma4:cloud com temperature=0
        self.llm = llm or ChatOllama(model="gemma4:cloud", temperature=0)

        self.documentos = []
        self.chunks = []
        self.vectorstore = None
        self.chain = None
        self._inicializar_pipeline()

    # ------------------------------------------------------------------ #
    # Indexação: load -> split -> embed -> store
    # ------------------------------------------------------------------ #
    def _criar_cliente_chroma(self):
        """PersistentClient grava no disco; Client() fica só em memória (Aula 05)."""
        if self.persist_dir:
            return chromadb.PersistentClient(path=self.persist_dir)
        return chromadb.Client()

    @staticmethod
    def _numerar_trechos(chunks: list):
        """Dá a cada chunk um número dentro do seu arquivo, usado na citação da fonte."""
        contadores = {}
        for chunk in chunks:
            arquivo = chunk.metadata.get("arquivo", "doc")
            contadores[arquivo] = contadores.get(arquivo, 0) + 1
            chunk.metadata["trecho"] = contadores[arquivo]

    def _assinatura(self, chunks: list) -> str:
        """Hash do conteúdo indexado: se mudar (docs/chunk_size), a coleção é recriada."""
        h = hashlib.sha256()
        h.update(f"{self.chunk_size}|{self.chunk_overlap}".encode())
        for c in chunks:
            h.update(c.page_content.encode())
            h.update(str(sorted(c.metadata.items())).encode())
        return h.hexdigest()

    def _indexar(self, chunks: list):
        """Cria a coleção no ChromaDB ou reaproveita a já salva em disco (Aula 05, Slide 16)."""
        self.client = self._criar_cliente_chroma()
        assinatura = self._assinatura(chunks)

        try:
            existente = self.client.get_collection(self.nome_colecao)
        except Exception:
            existente = None

        if existente is not None:
            if (existente.metadata or {}).get("assinatura") == assinatura and existente.count() > 0:
                # Mesmos documentos e chunking: não recalcula os embeddings
                return Chroma(
                    client=self.client,
                    collection_name=self.nome_colecao,
                    embedding_function=self.embeddings,
                )
            self.client.delete_collection(self.nome_colecao)

        banco = Chroma(
            client=self.client,
            collection_name=self.nome_colecao,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine", "assinatura": assinatura},
        )
        ids = [f"{c.metadata['arquivo']}::{c.metadata['trecho']}" for c in chunks]
        banco.add_documents(chunks, ids=ids)
        return banco

    def _inicializar_pipeline(self):
        """Executa load -> split -> embed -> store e monta a chain de geração."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            return

        self.documentos = carregar_documentos(self.docs_dir)
        if not self.documentos:
            return

        # Splitter com os separadores obrigatórios do PDF do professor
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = splitter.split_documents(self.documentos)
        self._numerar_trechos(self.chunks)

        # ChromaDB com coleção identificando o domínio
        self.vectorstore = self._indexar(self.chunks)

        # Etapa generate: prompt de grounding + LLM + parser (LCEL, Aula 06)
        prompt = ChatPromptTemplate.from_template(PROMPT_RAG_TEXTO)
        self.chain = prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    # Retrieve (com metadata filtering e reranking opcionais)
    # ------------------------------------------------------------------ #
    def _formatar_contexto(self, docs):
        """Formata os trechos recuperados com rótulo de arquivo e número do trecho."""
        partes = []
        for d in docs:
            arquivo = d.metadata.get("arquivo") or os.path.basename(d.metadata.get("source", "doc"))
            trecho = d.metadata.get("trecho", "?")
            produto = d.metadata.get("produto", "")
            pagina = f" — pág. {d.metadata['page'] + 1}" if isinstance(d.metadata.get("page"), int) else ""
            partes.append(f"[Trecho {trecho} — {arquivo} — {produto}{pagina}]\n{d.page_content}")
        return "\n\n---\n\n".join(partes)

    def _reordenar(self, consulta: str, resultados: list) -> list:
        """Reranking: o cross-encoder lê (consulta, trecho) e reordena os candidatos."""
        modelo = self.reranker or _obter_cross_encoder()
        notas = modelo.score([(consulta, doc.page_content) for doc, _ in resultados])
        ordenados = sorted(zip(resultados, notas), key=lambda par: par[1], reverse=True)
        return [(doc, 1 - dist, float(nota)) for (doc, dist), nota in ordenados[: self.k]]

    def recuperar(self, consulta: str, filtro: dict | None = None, usar_reranking: bool | None = None) -> list:
        """
        Busca os trechos mais relevantes. Devolve lista de (Document, similaridade, nota_rerank).
        filtro: where= do ChromaDB, aplicado ANTES da busca (ex.: {"categoria": "notebook"}).
        """
        if self.vectorstore is None:
            return []
        reranquear = self.usar_reranking if usar_reranking is None else usar_reranking
        k_busca = max(self.k_busca_ampla, self.k) if reranquear else self.k

        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k_busca, filter=filtro)
        if not resultados:
            return []

        if reranquear:
            return self._reordenar(consulta, resultados)
        # O score do Chroma é distância cosseno (menor = mais parecido): similaridade = 1 - distância
        return [(doc, 1 - dist, None) for doc, dist in resultados]

    # ------------------------------------------------------------------ #
    # Generate
    # ------------------------------------------------------------------ #
    def consultar(self, consulta: str, categoria: str | None = None, usar_reranking: bool | None = None) -> dict:
        """
        Pipeline completo. Devolve {"resposta", "fontes", "documentos"},
        em que "fontes" lista o chunk de origem de cada trecho usado na resposta.
        """
        if not self.chain:
            return {"resposta": "Base de conhecimento não indexada ou vazia.", "fontes": [], "documentos": []}

        encontrados = self.recuperar(consulta, montar_filtro(categoria=categoria), usar_reranking)
        if not encontrados:
            return {"resposta": MENSAGEM_SEM_RESPOSTA, "fontes": [], "documentos": []}

        docs = [doc for doc, _, _ in encontrados]
        resposta = self.chain.invoke({"contexto": self._formatar_contexto(docs), "pergunta": consulta})

        fontes = [
            {
                "arquivo": doc.metadata.get("arquivo", ""),
                "produto": doc.metadata.get("produto", ""),
                "categoria": doc.metadata.get("categoria", ""),
                "trecho": doc.metadata.get("trecho", "?"),
                "pagina": doc.metadata["page"] + 1 if isinstance(doc.metadata.get("page"), int) else None,
                "similaridade": round(sim, 3),
                "nota_rerank": None if nota is None else round(nota, 3),
                "texto": doc.page_content,
            }
            for doc, sim, nota in encontrados
        ]
        return {"resposta": resposta, "fontes": fontes, "documentos": docs}

    def buscar(self, consulta: str, categoria: str | None = None, usar_reranking: bool | None = None) -> str:
        """Função modular requisitada pelo professor para reaproveitamento no CKP03."""
        return self.consultar(consulta, categoria, usar_reranking)["resposta"]

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def categorias(self) -> list:
        """Categorias disponíveis na base, para filtrar por metadata."""
        return listar_categorias(self.documentos)

    def descartar_colecao(self):
        """Apaga a coleção (usado na avaliação para não acumular coleções de teste)."""
        if self.vectorstore is not None:
            self.client.delete_collection(self.nome_colecao)
            self.vectorstore = None
            self.chain = None


def formatar_fontes(fontes: list, tamanho_trecho: int = 140) -> str:
    """Texto legível com as fontes usadas na resposta (para terminal e Gradio)."""
    if not fontes:
        return ""
    linhas = ["Fontes consultadas:"]
    for f in fontes:
        nota = f" · rerank {f['nota_rerank']}" if f.get("nota_rerank") is not None else ""
        trecho = f["texto"].replace("\n", " ")[:tamanho_trecho]
        pagina = f"pág. {f['pagina']}, " if f.get("pagina") else ""
        linhas.append(
            f"- {f['arquivo']} ({pagina}trecho {f['trecho']}, {f['produto']}, "
            f"similaridade {f['similaridade']}{nota}): \"{trecho}...\""
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------- #
# Função modular para o CKP03 (@tool do agente)
# ---------------------------------------------------------------------- #
_servico_padrao = None


def obter_servico() -> SuporteRAGService:
    """
    Cria (uma única vez) o serviço RAG com a configuração vencedora do RAGAS
    e o índice persistido em chroma_db/. Reutilizado a cada chamada de buscar().
    """
    global _servico_padrao
    if _servico_padrao is None:
        _servico_padrao = SuporteRAGService(
            docs_dir=os.path.join(PASTA_PROJETO, "docs"),
            chunk_size=chunk_size_vencedor(),
            persist_dir=os.path.join(PASTA_PROJETO, "chroma_db"),
        )
    return _servico_padrao


def buscar(consulta: str, categoria: str | None = None, usar_reranking: bool | None = None) -> str:
    """
    Interface exigida pelo professor para o CKP03: buscar(consulta) -> resposta com fonte citada.

    Exemplo de uso como tool no CKP03:
        from langchain_core.tools import tool
        from app.rag_service import buscar

        @tool
        def consultar_manuais(consulta: str) -> str:
            \"\"\"Busca nos manuais de suporte técnico e responde citando a fonte.\"\"\"
            return buscar(consulta)
    """
    return obter_servico().buscar(consulta, categoria, usar_reranking)
