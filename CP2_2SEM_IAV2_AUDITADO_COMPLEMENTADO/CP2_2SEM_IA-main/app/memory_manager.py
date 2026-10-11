from langchain_classic.memory import ConversationBufferMemory


def criar_memoria():
   

    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )


def visualizar_memoria(memoria):
   

    return memoria.load_memory_variables({})


def limpar_memoria(memoria):
    

    memoria.clear()
