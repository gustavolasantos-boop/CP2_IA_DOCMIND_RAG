# Relatório de auditoria — CKP02 DocMind RAG

## Parecer

**Status: NÃO PRONTO PARA ENTREGA FINAL ainda.** A estrutura e o código atendem boa parte dos requisitos estáticos, mas o pacote fornecido não contém uma avaliação RAGAS executada nem um notebook exportado com saídas. Não se deve inventar ou preencher métricas estimadas.

## O que está implementado

- Cinco PDFs reais de fabricantes na pasta `docs/`, com catálogo e URLs de origem.
- Pipeline modular de carregamento, `RecursiveCharacterTextSplitter`, embeddings `nomic-embed-text`, ChromaDB, recuperação e geração com `gemma4:cloud`.
- Temperatura 0 no LLM do pipeline RAG.
- Citação de arquivo/trecho/página e resposta de fallback definida no prompt.
- Duas ou mais configurações de chunking; o script compara 256, 512 e 1024 por padrão.
- Oito perguntas de teste, mais que as cinco mínimas, e métricas RAGAS `faithfulness` e `answer_relevancy`.
- Metadata filtering, reranking cross-encoder e interface Gradio (diferenciais).
- README com integrantes/RMs e instruções para adicionar documentos.
- Arquivos Python compilam sintaticamente durante a auditoria estática.

## Pendências impeditivas para a entrega

1. **Executar a avaliação RAGAS com a chave real da Ollama Cloud.** O ZIP original não continha `resultados/vencedora.json`, `resultados/ragas_por_pergunta.csv` ou um resumo de resultados. Logo, ainda não há prova de `faithfulness >= 0,7`, nem comparação quantitativa real.
2. **Atualizar o README com resultados medidos.** Rodar `python -m app.avaliar_rag` e depois `python -m app.atualizar_readme`. Preencher a análise do grupo com a configuração vencedora, as métricas e a pior pergunta.
3. **Executar o notebook inteiro sem erros no Colab** com `OLLAMA_API_KEY` em Secrets. O notebook recebido tem 17 células de código, todas sem saída e com `execution_count` nulo; ele não é o export executado exigido pelo professor.
4. **Exportar/substituir o notebook final.** Baixar o `.ipynb` do Colab depois do Run All, com as saídas preservadas, e incluir esse arquivo no pacote final.
5. **Conferir a saída do checklist.** Executar `python -m app.verificar_entrega` depois dos passos acima e resolver todas as falhas relevantes. O checklist é auxiliar; a validação real do professor prevalece.

## Comparação com as aulas

- **Aula 05:** o uso de `OllamaEmbeddings`, ChromaDB, coleção persistente, busca por similaridade e filtros de metadados segue a arquitetura ensinada. A configuração foi ajustada para não oferecer uma opção de embeddings local, porque o CKP02 pede Ollama Cloud.
- **Aula 06:** o fluxo `load → split → embed → store → retrieve → generate`, PyMuPDF, `RecursiveCharacterTextSplitter`, chain LCEL e grounding com fonte estão implementados.
- **Aula 07:** há comparação de chunk sizes, RAGAS, diagnóstico da pergunta de menor faithfulness e diferencial de reranking. A comparação só conta como evidência de qualidade depois de executada com resultados reais.

## Comandos finais

Na raiz do projeto, com ambiente configurado e `OLLAMA_API_KEY` válida:

```bash
python -m app.avaliar_rag
python -m app.atualizar_readme
python -m app.verificar_entrega
```

No Colab, executar todas as células, verificar ausência de erros e exportar o notebook com saídas. Não incluir `.env` com credenciais no ZIP nem publicar a chave.
