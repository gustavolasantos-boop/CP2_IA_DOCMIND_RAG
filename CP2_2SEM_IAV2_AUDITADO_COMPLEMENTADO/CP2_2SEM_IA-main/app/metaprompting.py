"""Meta prompting: usa o próprio LLM para otimizar o system prompt.

Além de gerar a versão otimizada (técnica da Aula 04), este módulo também
imprime o "antes" (o SYSTEM_PROMPT original de app/prompts.py) lado a lado
com o "depois" gerado pelo modelo, e mede a redução de tokens com tiktoken —
documentando de forma objetiva o diferencial de context engineering.
"""

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .chains import llm


PROMPT_OTIMIZADOR = """<tarefa>
Você é especialista em context engineering.
Reescreva o system prompt recebido aplicando:
- XML tagging para separar seções;
- instruções críticas no início e no final;
- remoção de redundâncias;
- restrições claras e objetivas.
</tarefa>

<prompt_original>
{prompt_original}
</prompt_original>

<formato_saida>
Retorne o prompt otimizado entre as tags
<prompt_otimizado>...</prompt_otimizado>.
Depois explique em 3 bullet points as principais mudanças.
</formato_saida>"""


chain_otimizador = (
    ChatPromptTemplate.from_template(PROMPT_OTIMIZADOR)
    | llm
    | StrOutputParser()
)


def contar_tokens(texto: str) -> int:
    """Aproximação de contagem de tokens via tiktoken (mesma técnica da Aula 04)."""
    return len(tiktoken.encoding_for_model("gpt-4").encode(texto))


def executar_meta_prompting():
    from .prompts import SYSTEM_PROMPT

    resultado = chain_otimizador.invoke({"prompt_original": SYSTEM_PROMPT})

    tokens_antes = contar_tokens(SYSTEM_PROMPT)
  
    tokens_depois_resposta = contar_tokens(resultado)

    print("=== ANTES (system prompt atual, app/prompts.py) ===")
    print(SYSTEM_PROMPT)
    print(f"\nTokens (aprox., tiktoken): {tokens_antes}\n")

    print("=== DEPOIS (saída do meta prompting) ===")
    print(resultado)
    print(f"\nTokens da resposta completa do modelo (aprox.): {tokens_depois_resposta}")
    print(
        "\nObs.: a resposta do modelo inclui o prompt otimizado + a explicação "
        "das mudanças. Para comparar 'maçã com maçã', copie apenas o trecho "
        "entre <prompt_otimizado> e </prompt_otimizado> impresso acima e "
        "passe-o para contar_tokens() isoladamente."
    )


if __name__ == "__main__":
    executar_meta_prompting()