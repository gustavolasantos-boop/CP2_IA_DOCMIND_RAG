import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationChain
from langchain_core.output_parsers import PydanticOutputParser

from .memory_manager import criar_memoria
from .prompts import prompt_chat, prompt_analise
from .schemas import AnaliseAtendimento


load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

if not api_key:
    raise ValueError(
        "OLLAMA_API_KEY não encontrada. Crie um arquivo .env na raiz do projeto."
    )


os.environ["OLLAMA_HOST"] = "https://ollama.com"
os.environ["OLLAMA_API_KEY"] = api_key


llm = ChatOllama(
    model="gemma4:cloud",
    temperature=0.7,
    num_predict=700,
)


memoria = criar_memoria()

chat_chain = ConversationChain(
    llm=llm,
    memory=memoria,
    prompt=prompt_chat,
    verbose=False,
)


parser = PydanticOutputParser(pydantic_object=AnaliseAtendimento)

analise_chain = (
    prompt_analise.partial(
        format_instructions=parser.get_format_instructions()
    )
    | llm
    | parser
)


def analisar_atendimento(pergunta, resposta):
    """Converte uma resposta livre em um objeto Pydantic validado."""

    return analise_chain.invoke({
        "pergunta": pergunta,
        "resposta": resposta,
    })

