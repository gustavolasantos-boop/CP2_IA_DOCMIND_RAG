# Projeto Chatbot Profissional · Suporte Técnico de Aparelhos Eletrônicos

Prompt Engineering & AI · FIAP · 2º Semestre 2026

## Integrantes

- Gabriel Camarosani Gouvea Gonçalves da Silva — RM 569189
- Gustavo Lima Andrade Santos — RM 571709
- Lucas Seiji Hummel — RM 569673
- Pedro Souza Castro — RM 569311
- Bruno Yudi Moritaka Kanashiro — RM 571776
- Lucas Barreto Santana — RM 573149

---

# CKP02 — DocMind RAG · Pipeline RAG completo com ChromaDB e RAGAS

## Domínio e base de conhecimento

O domínio é o mesmo do CKP01: suporte técnico de aparelhos eletrônicos. A base tem 5 documentos reais, na pasta `docs/`, em 3 categorias (notebook, roteador e TV). Os cinco são PDFs oficiais dos fabricantes, sem resumos nem edições. Os metadados de cada documento ficam em `docs/catalogo.json`.

| Documento | Produto | Categoria | Tipo | Fonte |
|---|---|---|---|---|
| `dell-g15-5530-owners-manual-pt-br.txt` | Dell G15 5530 | notebook | manual | https://www.dell.com/support/product-details/pt-br/product/g-series-15-5530-laptop/resources/manuals |
| `intelbras-rf-301k-manual.pdf` | Intelbras RF 301K | roteador | manual | https://backend.intelbras.com/sites/default/files/2025-03/Manual_RF_301K_portugues_01-25_site.pdf |
| `intelbras-wrn-240-manual.pdf` | Intelbras WRN 240 | roteador | manual | https://backend.intelbras.com/sites/default/files/integration/manual_wrn240_01-12_site.pdf |
| `tp-link-archer-c60-guia-rapido.pdf` | TP-Link Archer C60 (BR) | roteador | guia rápido | https://static.tp-link.com/upload/manual/2022/202201/20220107/7106509419_Archer%20C60(BR)_QIG_V1.pdf |
| `philips-smart-tv-pfl-manual.pdf` | Philips Smart TV (série PFL) | tv | manual | https://www.documents.philips.com/assets/20230308/127083b0a2364557ab74afbf01014ce4.pdf |

Os PDFs foram baixados dos links acima e estão em `docs/` com os nomes da tabela.

## Pipeline RAG

```text
docs/ (TXT, MD ou PDF) + docs/catalogo.json
        |  LOAD    base_conhecimento.py  (metadados: produto, categoria, tipo, fonte, data)
        v
RecursiveCharacterTextSplitter
separators = ["\n\n", "\n", ". ", " ", ""]   chunk_overlap = chunk_size // 8
        |  SPLIT
        v
OllamaEmbeddings(model="nomic-embed-text")
        |  EMBED
        v
ChromaDB (chromadb.PersistentClient / Client)  coleção "suporte_eletronicos_<chunk_size>", métrica cosseno
        |  STORE
        v
Busca por similaridade (k=3)  [where= opcional]  [cross-encoder opcional: k=10 -> top 3]
        |  RETRIEVE
        v
Prompt de grounding + ChatOllama("gemma4:cloud", temperature=0)
        |  GENERATE
        v
Resposta com citação (fonte: arquivo, trecho N, pág. P) + lista dos chunks de origem
```

Configuração: `OllamaEmbeddings(nomic-embed-text)`, `gemma4:cloud` com temperatura 0, `k=3` e overlap de `chunk_size // 8` (12,5% do chunk, dentro da faixa de 10 a 15%).

O código está em `app/rag_service.py`, classe `SuporteRAGService`. Para o CKP03, a função modular é `buscar(consulta)` (função de módulo em `app/rag_service.py`, que também existe como método do serviço), e devolve a resposta em texto. Ela cria o serviço uma única vez com o `chunk_size` vencedor do RAGAS e o índice persistido em `chroma_db/`. `consultar(consulta)` devolve também os chunks de origem (arquivo, trecho, similaridade e texto).

```python
from app.rag_service import buscar, SuporteRAGService

print(buscar("Como restaurar o roteador Intelbras RF 301K para a configuração de fábrica?"))

# Controle total (chunk_size, filtro, reranking e chunks de origem):
rag = SuporteRAGService(docs_dir="docs", chunk_size=1024)
resultado = rag.consultar("Como restaurar as configurações de fábrica?", categoria="roteador", usar_reranking=True)
```

Uso como tool no CKP03:

```python
from langchain_core.tools import tool
from app.rag_service import buscar

@tool
def consultar_manuais(consulta: str) -> str:
    """Busca nos manuais de suporte técnico e responde citando a fonte."""
    return buscar(consulta)
```

### Citação da fonte

Cada trecho enviado ao LLM recebe um rótulo `[Trecho N — arquivo — produto — pág. P]` (a página só aparece nos PDFs), e o prompt exige a citação `(fonte: arquivo, trecho N, pág. P)`. O terminal e a interface Gradio também listam os chunks usados, com a similaridade de cada um.

### Diferenciais

- **Metadata filtering:** cada documento tem `produto`, `marca`, `categoria`, `tipo`, `formato_original`, `fonte_url` e `data_inclusao`. A busca aceita `where=` pelo parâmetro `categoria` (ou `montar_filtro(categoria=..., marca=...)`, que gera o `$and`). O filtro restringe o espaço de busca antes do k-NN.
- **Reranking:** busca ampla (k=10) e reordenação com `cross-encoder/ms-marco-MiniLM-L-6-v2`, ficando os 3 melhores. Ative com `usar_reranking=True` no serviço, no comando `rerank` do chatbot ou na caixa da interface.
- **Interface Gradio:** `python -m app.interface_gradio` abre um `gr.ChatInterface` com o RAG integrado, filtro de categoria, reranking e fontes citadas na resposta.

## Avaliação com RAGAS

O script `app/avaliar_rag.py` usa o mesmo pipeline do chatbot e compara configurações de `chunk_size` (por padrão 256, 512 e 1024, as mesmas da demo da Aula 07; aceita outras com `--chunks`). O teste usa oito perguntas (`app/perguntas_teste.py`), sendo sete dentro dos manuais, cobrindo os cinco documentos, e uma sobre um assunto que não está nos documentos (geladeira), para testar a recusa. Cada pergunta registra o documento, a página do PDF e a resposta esperada, para conferência manual.

Métricas:

- **Faithfulness:** avalia se a resposta está apoiada no contexto recuperado. Meta do CKP02: 0,7 ou mais (aprovação) e 0,9 ou mais (zona ideal).
- **Answer relevancy:** avalia a relação entre a resposta e a pergunta.

A cada execução o script grava em `resultados/`: `ragas_por_pergunta.csv` (tabela por pergunta e por configuração), `ragas_resumo.md` (médias e configuração vencedora) e `vencedora.json`. O `vencedora.json` é lido por `app/config_rag.py`, então o chatbot (`main.py`), a interface Gradio e o `buscar()` do CKP03 passam a usar automaticamente o `chunk_size` vencedor. A vencedora é a de maior faithfulness, com answer relevancy como desempate.

## Resultados da execução

**Status antes da entrega:** os resultados RAGAS não são incluídos previamente, porque precisam ser medidos com a chave da Ollama Cloud e não podem ser inventados. Portanto, este pacote ainda NÃO deve ser enviado como entrega final até executar a avaliação, atualizar esta seção e exportar o notebook com as saídas.

Para gerar os resultados reais, execute `python -m app.avaliar_rag` e depois `python -m app.atualizar_readme` (ou execute as seções 11 e 12 do notebook e atualize o README).

<!-- RESULTADOS_RAGAS_INICIO -->
_PENDENTE: execute a avaliação RAGAS com Ollama Cloud. A tabela por pergunta, as médias e a configuração vencedora devem ser geradas a partir da execução real._
<!-- RESULTADOS_RAGAS_FIM -->

## Análise dos resultados

> **Preencher após a execução** (o critério do professor exige justificar a escolha com os dados): qual `chunk_size` venceu e por que (hipótese do grupo), se o faithfulness médio ficou em 0,7 ou mais, quais perguntas tiveram nota baixa e por quê (use a análise de falhas da seção 12.1 do notebook). Chatbot, Gradio e `buscar()` já usam a configuração vencedora automaticamente.

## Como adicionar novos documentos à base

1. Coloque o arquivo na pasta `docs/`. São aceitos `.txt`, `.md` e `.pdf` (PDFs são lidos com PyMuPDF, uma página por documento).
2. Registre o documento em `docs/catalogo.json`, copiando um item existente e ajustando os campos:

   ```json
   {
     "arquivo": "manual-novo-aparelho.pdf",
     "produto": "Nome do produto",
     "marca": "Marca",
     "categoria": "notebook",
     "tipo": "manual",
     "formato_original": "pdf",
     "fonte_url": "https://link-publico-do-documento",
     "data_inclusao": "2026-10-10"
   }
   ```

   Sem esse registro o documento ainda é indexado, mas entra com `categoria = "geral"` e sem fonte.
3. Registre a origem na tabela de fontes deste README.
4. Rode de novo o chatbot, a interface ou a avaliação. O índice do ChromaDB (`chroma_db/`) é recriado sozinho quando os documentos ou o `chunk_size` mudam. Se quiser forçar uma reindexação, apague a pasta `chroma_db/`.
5. Para a avaliação, acrescente perguntas em `app/perguntas_teste.py` se o novo documento trouxer assuntos novos.

## Entrega no Colab

O notebook `CKP02_suporte_eletronicos_grupo02.ipynb` (padrão `CKP02_[dominio]_grupo.ipynb`) percorre todas as etapas (load, split, embed, store, retrieve, generate, filtro, reranking, RAGAS com análise de falhas, `buscar()` e Gradio) usando os módulos de `app/`.

Roteiro de entrega:

1. Abra o notebook no Colab e salve `OLLAMA_API_KEY` em *Secrets*.
2. Envie o `.zip` do projeto quando pedido e use *Run All* (leva alguns minutos por causa do RAGAS).
3. Preencha a célula 12.3 (análise do grupo: qual configuração ganhou e por quê).
4. Rode `python -m app.atualizar_readme` (ou copie a tabela da seção 12.2) para preencher o README com os números reais.
5. Rode `python -m app.verificar_entrega` (ou a última célula do notebook): confere os requisitos do PDF do professor. O checklist deve ser revisado após a avaliação real.
6. Depois de executar todas as células sem erros, baixe/exporte o `.ipynb` **já executado, com as saídas**, e substitua a cópia não executada do pacote por esse arquivo final.
7. Pelo Teams, somente o líder envia: link do Colab, `.ipynb` executado, `README.md` e a pasta `docs/`, até 23:59 do dia da Aula 08.

---

# CKP01 — Chatbot Profissional

## Domínio

O chatbot atua no domínio de suporte técnico de aparelhos eletrônicos.

O objetivo é auxiliar o usuário na identificação de problemas comuns de hardware, software e configuração, fornecendo orientações práticas e seguras.

Usuários-alvo: pessoas que precisam de uma primeira orientação sobre problemas em aparelhos eletrônicos e querem entender possíveis causas e próximos passos.

O domínio foi mantido restrito para que o system prompt, a memória e a saída estruturada tenham uma finalidade clara. O mesmo domínio pode ser reutilizado nos próximos checkpoints do semestre.

## Arquitetura

O projeto segue a arquitetura de duas chains trabalhada na Aula 03:

```text
Pergunta do usuário
       |
       v
ConversationChain
+ ConversationBufferMemory
       |
       v
Resposta textual
       |
       +----------------------+
       |                      |
       v                      v
Histórico da sessão     Chain LCEL de análise
                        ChatPromptTemplate
                                |
                                v
                           ChatOllama
                                |
                                v
                      PydanticOutputParser
                                |
                                v
                        AnaliseAtendimento
```

### Chain 1 — conversa

`ConversationChain` recebe a pergunta e mantém o histórico em `ConversationBufferMemory`.

A memória é de curto prazo, existe durante a execução do programa e é apagada quando a aplicação é encerrada.

### Chain 2 — saída estruturada

A resposta produzida pelo chatbot é enviada para uma segunda chain:

```text
ChatPromptTemplate | ChatOllama | PydanticOutputParser
```

O resultado é um objeto `AnaliseAtendimento`, e não uma string livre.

## Justificativa da memória

Foi escolhido `ConversationBufferMemory` porque o caso de uso é um atendimento técnico por sessão, no qual as informações dos turnos anteriores podem ser importantes para entender sintomas, aparelho e tentativas já realizadas.

A principal vantagem é a fidelidade: o histórico completo fica disponível para os próximos turnos. A desvantagem é que o custo de tokens cresce conforme a conversa fica maior.

Essa escolha é adequada para uma sessão curta de suporte. Em uma conversa muito longa, uma estratégia Summary ou TokenBuffer poderia reduzir o crescimento do contexto.

## Demonstração de 5+ turnos

Após iniciar:

```bash
python -m app.main
```

use uma sequência semelhante:

```text
Você: Meu notebook está desligando sozinho.
Você: Isso acontece quando estou usando jogos.
Você: A temperatura parece alta.
Você: O problema começou ontem.
Você: Já limpei as entradas de ar.
Você: Qual era o problema que eu mencionei no início?
```

O último turno depende das informações anteriores e demonstra a memória conversacional.

Também é possível digitar:

```text
memoria
```

para visualizar o histórico atual, ou:

```text
limpar
```

para apagá-lo.

### Demonstração automática (sem digitação manual)

Para uma evidência objetiva e reprodutível do requisito "memória funciona em 5+ turnos", também é possível rodar:

```bash
python -m app.demo_memoria
```

Esse script executa 6 turnos fixos automaticamente (sem precisar digitar nada) e imprime, ao final, o estado bruto da memória via `load_memory_variables({})`, confirmando que o último turno — que pergunta "qual foi o aparelho mencionado no início?" — é respondido corretamente com base no primeiro turno da conversa.

## Pydantic v2 e validação

O schema `AnaliseAtendimento` possui seis campos:

- `categoria: str`
- `componente: str`
- `prioridade: Literal["baixa", "media", "alta"]`
- `resumo: str`
- `passos: List[str]`
- `precisa_tecnico: bool`

O `PydanticOutputParser` recebe esse schema e gera as instruções de formato que são inseridas no prompt por `.partial()`.

Depois da resposta do modelo, o parser valida os dados.

A aplicação também utiliza `try/except ValidationError` para evitar que uma saída inválida derrube o programa.

## Context Engineering

O system prompt foi organizado com XML tagging:

- `<persona>`
- `<dominio>`
- `<restricoes>`
- `<formato>`
- `<seguranca>`
- `<regra_final>`

As instruções críticas aparecem no início e no final do contexto. As mensagens do usuário são tratadas como dados e não podem substituir as regras do sistema.

O objetivo é aplicar o conceito de context engineering: selecionar e organizar intencionalmente o que entra na janela de contexto, evitando informação desnecessária.

## Context rot

O arquivo `app/context_rot.py` executa um experimento usando o mesmo modelo e a mesma pergunta final com diferentes tamanhos de contexto:

- 0 turnos
- 5 turnos
- 10 turnos
- 15 turnos
- 20 turnos

A instrução-alvo é colocada no meio do histórico e existem mensagens de distração ao redor dela.

O programa mede uma aproximação do número de tokens com `tiktoken` e registra se a resposta preservou a instrução-alvo.

Execute:

```bash
python -m app.context_rot
```

O resultado é gerado pelo modelo no momento da execução. Assim, a tabela apresentada pelo programa representa o teste executado naquele ambiente, em vez de uma pontuação inventada previamente.

`tiktoken` é utilizado como aproximação para a contagem, conforme apresentado na Aula 04; ele não representa necessariamente a contagem interna exata do `gemma4:cloud`.

## Meta prompting

O arquivo `app/metaprompting.py` aplica a técnica apresentada na Aula 04: o próprio modelo recebe o system prompt atual e produz uma versão otimizada.

Execute:

```bash
python -m app.metaprompting
```

A saída documenta o antes/depois de forma explícita:

- O `SYSTEM_PROMPT` original (o "antes"), com sua contagem aproximada de tokens via `tiktoken`.
- O prompt otimizado retornado pelo modelo entre `<prompt_otimizado>...</prompt_otimizado>` (o "depois"), seguido de 3 bullet points explicando as mudanças e da contagem de tokens da resposta.

Isso deixa o diferencial "meta prompting: usar o modelo para melhorar o prompt e documentar o antes/depois" evidenciado diretamente na execução, sem depender só da leitura do código.

## Segurança

O system prompt possui regras contra:

- prompt injection;
- jailbreak;
- tentativa de extração do system prompt;
- exposição de chaves e credenciais;
- instruções perigosas envolvendo hardware ou eletricidade.

A segurança foi implementada usando as técnicas trabalhadas no módulo: regras explícitas, separação entre dados e instruções e organização do contexto com XML.

---

# Estrutura do projeto

```text
CKP02-2SEM-DocMind-RAG_Grupo02/
├── .env.example               (modelo; não contém a chave real)
├── .gitignore
├── README.md
├── requirements.txt
├── CKP02_suporte_eletronicos_grupo02.ipynb
├── resultados/                 (gerada por app/avaliar_rag.py: CSV, resumo .md e vencedora.json)
├── chroma_db/                  (gerada em execução; índice do ChromaDB)
├── docs/
│   ├── catalogo.json
│   ├── dell-g15-5530-owners-manual-pt-br.pdf
│   ├── intelbras-rf-301k-manual.pdf
│   ├── intelbras-wrn-240-manual.pdf
│   ├── tp-link-archer-c60-guia-rapido.pdf
│   └── philips-smart-tv-pfl-manual.pdf
└── app/
    ├── __init__.py
    ├── main.py
    ├── chains.py
    ├── memory_manager.py
    ├── schemas.py
    ├── context_rot.py
    ├── metaprompting.py
    ├── demo_memoria.py
    ├── prompts.py
    ├── base_conhecimento.py
    ├── rag_service.py
    ├── interface_gradio.py
    ├── config_rag.py           (chunk_size vencedor compartilhado)
    ├── perguntas_teste.py
    ├── avaliar_rag.py
    ├── atualizar_readme.py     (preenche o README com os resultados reais)
    └── verificar_entrega.py    (checklist da rubrica do CKP02)
```

# Como executar

### 1. Criar o ambiente

Na pasta raiz:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar a Ollama Cloud

Crie `.env` a partir do exemplo:

```powershell
copy .env.example .env
```

Edite `.env`:

```text
OLLAMA_API_KEY=sua_chave_da_ollama_cloud
```

Nunca envie o `.env` para o GitHub ou para o arquivo `.zip`.

### 4. Executar o chatbot

```bash
python -m app.main
```

### 5. Executar a demonstração automática de memória (>= 5 turnos)

```bash
python -m app.demo_memoria
```

### 6. Executar o teste de context rot

```bash
python -m app.context_rot
```

### 7. Executar o meta prompting

```bash
python -m app.metaprompting
```

### 8. Executar a avaliação do RAG (CKP02)

Na pasta raiz do projeto:

```bash
python -m app.avaliar_rag
```

Opções: `--chunks 512 1024` para comparar menos configurações (padrão: 256 512 1024) e `--rerank` para medir com reranking. No PyCharm também funciona `python app/avaliar_rag.py`.

### 9. Executar a interface Gradio (CKP02)

```bash
python -m app.interface_gradio
```

Para gerar um link público temporário, defina `GRADIO_SHARE=1` antes de executar.

No chatbot de terminal (`python -m app.main`), use `rag: <pergunta>` para consultar a base, `filtro <categoria>` para filtrar por metadata (ex.: `filtro roteador`; `filtro todas` limpa) e `rerank` para ligar ou desligar o reranking.

### 10. Atualizar o README e conferir a entrega (CKP02)

```bash
python -m app.atualizar_readme    # preenche "Resultados da execução" com os números reais
python -m app.verificar_entrega   # checklist dos requisitos do professor
```

# Dependências

As dependências estão em `requirements.txt` e correspondem às bibliotecas utilizadas no projeto:

- LangChain Ollama
- LangChain Core
- LangChain Classic
- Pydantic v2
- python-dotenv
- tiktoken
- LangChain Community
- LangChain Text Splitters
- ChromaDB
- RAGAS
- PyMuPDF (leitura de PDFs)
- sentence-transformers (cross-encoder do reranking)
- Gradio (interface do RAG)
- pandas e datasets (tabelas e dataset do RAGAS)

# Modelo utilizado

O projeto utiliza:

```text
gemma4:cloud
```

por meio da Ollama Cloud, com a chave armazenada em variável de ambiente.

Para os embeddings, o projeto utiliza exclusivamente `nomic-embed-text` via Ollama Cloud, conforme o requisito do CKP02. Não configure um provedor local ou outro modelo de embedding para a entrega.
