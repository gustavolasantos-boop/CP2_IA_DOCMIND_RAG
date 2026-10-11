from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class AnaliseAtendimento(BaseModel):
    """Schema estruturado de uma interação de suporte técnico."""

    categoria: str = Field(
        description="Categoria do problema, por exemplo hardware, software ou configuração."
    )
    componente: str = Field(
        description="Componente ou parte do aparelho relacionada ao problema."
    )
    prioridade: Literal["baixa", "media", "alta"] = Field(
        description="Prioridade estimada do atendimento."
    )
    resumo: str = Field(
        description="Resumo objetivo do problema e da orientação dada."
    )
    passos: List[str] = Field(
        description="Passos de orientação presentes na resposta."
    )
    precisa_tecnico: bool = Field(
        description="Indica se a resposta recomenda assistência técnica qualificada."
    )

    @field_validator("passos")
    @classmethod
    def validar_passos(cls, valor):
        if len(valor) == 0:
            raise ValueError("passos deve conter pelo menos um item")
        return valor