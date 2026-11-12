import os

from dotenv import load_dotenv

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

_FALLBACK_RESPONSE = "Não tenho informações necessárias para responder sua pergunta."
_SEARCH_CHAIN = None

def _required_settings():
    load_dotenv()

    settings = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_EMBEDDING_MODEL": os.getenv("GOOGLE_EMBEDDING_MODEL"),
        "GOOGLE_CHAT_MODEL": os.getenv("GOOGLE_CHAT_MODEL"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "PG_VECTOR_COLLECTION_NAME": os.getenv("PG_VECTOR_COLLECTION_NAME"),
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(
            "Variáveis de ambiente obrigatórias não configuradas: "
            + ", ".join(missing)
        )
    return settings

def _document_context(documents):
    context_parts = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source")
        page = metadata.get("page")

        provenance = []
        if source:
            provenance.append(f"Fonte: {source}")
        if page is not None:
            provenance.append(f"Página: {page}")

        header = " | ".join(provenance)
        content = getattr(document, "page_content", "")
        if not content:
            continue
        context_parts.append(f"{header}\n{content}" if header else content)

    return "\n\n---\n\n".join(context_parts)


class _SearchChain:
    def __init__(self, vector_store, chat_model):
        self.vector_store = vector_store
        self.chat_model = chat_model

    def invoke(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A pergunta deve ser uma string não vazia.")

        documents_with_scores = self.vector_store.similarity_search_with_score(
            question.strip(), k=10
        )
        documents = [document for document, _score in documents_with_scores]
        context = _document_context(documents)
        if not context:
            return _FALLBACK_RESPONSE

        prompt = PROMPT_TEMPLATE.format(
            contexto=context,
            pergunta=question.strip(),
        )
        response = self.chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str) and content.strip():
            return content.strip()
        return _FALLBACK_RESPONSE

def _build_search_chain():
    settings = _required_settings()

    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from langchain_postgres import PGVector

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings["GOOGLE_EMBEDDING_MODEL"],
            google_api_key=settings["GOOGLE_API_KEY"],
        )
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=settings["PG_VECTOR_COLLECTION_NAME"],
            connection=settings["DATABASE_URL"],
            use_jsonb=True,
        )
        chat_model = ChatGoogleGenerativeAI(
            model=settings["GOOGLE_CHAT_MODEL"],
            google_api_key=settings["GOOGLE_API_KEY"],
            temperature=0,
        )
    except Exception as error:
        raise RuntimeError(
            f"Não foi possível inicializar a busca ({type(error).__name__})."
        ) from error

    return _SearchChain(vector_store, chat_model)

def search_prompt(question=None):
    global _SEARCH_CHAIN

    if _SEARCH_CHAIN is None:
        try:
            _SEARCH_CHAIN = _build_search_chain()
        except (RuntimeError, ValueError):
            return None

    if question is None:
        return _SEARCH_CHAIN
    return _SEARCH_CHAIN.invoke(question)
