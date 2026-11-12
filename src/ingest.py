import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def ingest_pdf():
    """Carregamento do PDF, divisão de seu conteúdo e persistência dos vetores no PostgreSQL."""
    pdf_path = os.getenv("PDF_PATH")
    database_url = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    required_variables = {
        "PDF_PATH": pdf_path,
        "DATABASE_URL": database_url,
        "PG_VECTOR_COLLECTION_NAME": collection_name,
        "GOOGLE_EMBEDDING_MODEL": embedding_model,
        "GOOGLE_API_KEY": google_api_key,
    }
    missing_variables = [
        name for name, value in required_variables.items() if not value
    ]
    if missing_variables:
        print(
            "Erro: variáveis de ambiente obrigatórias não configuradas: "
            + ", ".join(missing_variables)
        )
        return

    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        print(f"Erro: o arquivo PDF não foi encontrado: {pdf_file}")
        return

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_postgres import PGVector
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        print(f"Páginas carregadas: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Chunks gerados: {len(chunks)}")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key,
        )
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=database_url,
            use_jsonb=True,
        )
        vector_store.add_documents(chunks)

        print(
            "Ingestão concluída com sucesso: "
            f"{len(chunks)} chunks armazenados na coleção "
            f"'{collection_name}'."
        )
    except Exception as error:
        print(f"Erro durante a ingestão: {type(error).__name__}")
        print(
            "Verifique a API Gemini, as credenciais, o PDF e a conexão com o PostgreSQL."
        )

if __name__ == "__main__":
    ingest_pdf()
