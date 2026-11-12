from search import search_prompt

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat RAG iniciado. Digite sua pergunta ou 'sair' para encerrar.")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChat encerrado.")
            return

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("Chat encerrado.")
            return

        try:
            resposta = chain.invoke(pergunta)
            print(resposta)
        except Exception:
            print("Não foi possível processar sua pergunta. Tente novamente.")

if __name__ == "__main__":
    main()
