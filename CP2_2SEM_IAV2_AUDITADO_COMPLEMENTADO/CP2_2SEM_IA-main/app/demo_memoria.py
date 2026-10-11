"""Demonstração automática e reprodutível da memória conversacional.

Este script existe para tornar o requisito R2 do CKP01 ("Demonstração de que
funciona em >= 5 turnos"). Ele roda uma sequência fixa de 6 turnos
sobre o domínio do grupo (suporte técnico de aparelhos eletrônicos), imprime
a resposta de cada turno e, ao final, imprime o estado bruto da memória com
load_memory_variables({}).

Execute com:
    python -m app.demo_memoria
"""

from .chains import chat_chain
from .memory_manager import visualizar_memoria, limpar_memoria


TURNOS_DEMO = [
    "Meu notebook está desligando sozinho quando eu jogo.",
    "Isso acontece depois de uns 20 minutos de uso.",
    "A parte de baixo do notebook fica bem quente.",
    "Já limpei as entradas de ar com ar comprimido.",
    "O problema começou a acontecer ontem, do nada.",
    "Qual foi o aparelho que eu mencionei logo no início da nossa conversa?",
]


def executar_demo():
 
    limpar_memoria(chat_chain.memory)

    print("=== DEMONSTRAÇÃO AUTOMÁTICA DE MEMÓRIA (>= 5 TURNOS) ===\n")

    for i, pergunta in enumerate(TURNOS_DEMO, start=1):
        resposta = chat_chain.predict(input=pergunta)
        print(f"Turno {i}")
        print(f"  Você: {pergunta}")
        print(f"  Assistente: {resposta}\n")

    print("=== ESTADO FINAL DA MEMÓRIA (load_memory_variables) ===")
    estado = visualizar_memoria(chat_chain.memory)
    print(estado)
    print(f"\nTotal de mensagens guardadas: {len(estado['history'])}")
    print(
        "Se o último turno citou corretamente o notebook/jogo do turno 1, "
        "a memória conversacional está funcionando."
    )


if __name__ == "__main__":
    executar_demo()