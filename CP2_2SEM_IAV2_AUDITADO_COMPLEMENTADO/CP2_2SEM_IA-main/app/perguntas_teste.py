"""
Perguntas de teste do CKP02 (usadas por app/avaliar_rag.py e pelo notebook).

Cada pergunta traz o documento e a página (do PDF) onde a resposta está, e um
resumo da resposta esperada, para o grupo conferir manualmente as respostas do RAG.
A última pergunta está FORA da base de propósito: o correto é o sistema recusar.
"""

TESTES = [
    {
        "pergunta": "Como forçar o desligamento do notebook Dell G15 quando ele travar?",
        "documento": "dell-g15-5530-owners-manual-pt-br.pdf",
        "pagina_pdf": 8,
        "esperado": "Manter pressionado o botão liga/desliga por 10 segundos para forçar o desligamento.",
    },
    {
        "pergunta": "Como restaurar o roteador Intelbras RF 301K para as configurações de fábrica?",
        "documento": "intelbras-rf-301k-manual.pdf",
        "pagina_pdf": 25,
        "esperado": "Pressionar o botão WPS/RST por 20 segundos e configurar de novo pelo assistente.",
    },
    {
        "pergunta": "Qual é o usuário e a senha padrão para acessar a configuração do roteador Intelbras WRN 240?",
        "documento": "intelbras-wrn-240-manual.pdf",
        "pagina_pdf": 7,
        "esperado": "Usuário admin e senha admin.",
    },
    {
        "pergunta": "Como conectar um dispositivo à rede Wi-Fi do Intelbras WRN 240 usando o botão WPS?",
        "documento": "intelbras-wrn-240-manual.pdf",
        "pagina_pdf": 31,
        "esperado": "Clicar uma vez em WPS no roteador e, no adaptador cliente, pressionar WPS por 6 s; "
                    "a associação é automática.",
    },
    {
        "pergunta": "O que fazer se eu esquecer a senha da rede Wi-Fi do roteador TP-Link?",
        "documento": "tp-link-archer-c60-guia-rapido.pdf",
        "pagina_pdf": 2,
        "esperado": "Ver a senha padrão na etiqueta embaixo do roteador, ou ligar por cabo, entrar em "
                    "http://tplinkwifi.net e ir em Básico > Wireless.",
    },
    {
        "pergunta": "A TV Philips está ligada, mas a imagem não aparece ou está distorcida. O que verificar?",
        "documento": "philips-smart-tv-pfl-manual.pdf",
        "pagina_pdf": 64,
        "esperado": "Verificar a antena, se a fonte (dispositivo) correta está selecionada e se o "
                    "dispositivo externo está bem conectado.",
    },
    {
        "pergunta": "Esqueci o código de 4 dígitos do bloqueio de canais da TV Philips. Qual código devo inserir?",
        "documento": "philips-smart-tv-pfl-manual.pdf",
        "pagina_pdf": 63,
        "esperado": "Inserir 8888.",
    },
    {
        "pergunta": "Qual é o procedimento para consertar o motor de uma geladeira?",  # fora do domínio
        "documento": None,
        "pagina_pdf": None,
        "esperado": "Recusar: 'Não encontrei essa informação nos documentos fornecidos.'",
    },
]

PERGUNTAS = [t["pergunta"] for t in TESTES]
