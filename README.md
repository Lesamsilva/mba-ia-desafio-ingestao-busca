# Ingestão e Busca Semântica com LangChain e PostgreSQL

Projeto desenvolvido como parte do desafio acadêmico de pós-graduação de ingestão e busca semântica de documentos, utilizando Python, LangChain, Google Gemini e PostgreSQL com a extensão pgVector.

## 1. Objetivo

Implementar uma aplicação baseada na arquitetura **RAG (Retrieval-Augmented Generation)**, capaz de responder perguntas em linguagem natural utilizando exclusivamente informações recuperadas de um documento PDF.

A aplicação possui duas etapas principais:

- **Ingestão:** leitura do PDF, divisão em chunks, geração de embeddings e persistência em banco vetorial.
- **Consulta:** recebimento de perguntas via terminal, busca semântica no banco vetorial e geração de respostas com o Google Gemini.

O sistema utiliza o conteúdo recuperado do PDF como contexto para a geração das respostas, seguindo instruções para não utilizar conhecimento externo.

## 2. Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3.11
| LangChain | Integração dos componentes de ingestão, recuperação e geração |
| Google Gemini | Geração de embeddings e respostas |
| PostgreSQL 17 | Banco de dados |
| pgVector | Armazenamento e busca vetorial |
| Docker e Docker Compose | Execução do banco de dados |
| PyPDFLoader | Leitura do documento PDF |
| RecursiveCharacterTextSplitter | Divisão do documento em chunks |
| langchain-postgres | Integração com PostgreSQL e pgVector |

### Modelos configurados

- **Embedding:** `models/gemini-embedding-001`
- **LLM:** `gemini-3.8-flash`

Os modelos são configurados por variáveis de ambiente. A disponibilidade dos modelos depende do provedor e pode mudar.

## 3. Arquitetura da solução

O processamento é dividido em dois fluxos independentes.

### 3.1. Ingestão

```text
document.pdf
     |
     v
PyPDFLoader
     |
     v
RecursiveCharacterTextSplitter
     |
     v
GoogleGenerativeAIEmbeddings
     |
     v
PostgreSQL + pgVector
```

O documento é dividido em chunks utilizando os parâmetros:

- `chunk_size = 1000`
- `chunk_overlap = 150`

Cada chunk é convertido em um embedding e armazenado na coleção vetorial configurada.

Na execução utilizada para validar o projeto, o documento de 34 páginas resultou em 67 chunks, com embeddings de 3072 dimensões.

### 3.2. Consulta semântica

```text
Pergunta do usuário
        |
        v
     chat.py
        |
        v
     search.py
        |
        v
Embedding da pergunta
        |
        v
Busca semântica no pgVector
        |
        v
10 chunks mais relevantes
        |
        v
Construção do prompt
        |
        v
Google Gemini
        |
        v
Resposta no terminal
```

Para cada pergunta, a aplicação utiliza:

```python
similarity_search_with_score(question, k=10)
```

Os dez resultados recuperados são organizados em um contexto textual e enviados ao modelo de linguagem junto com a pergunta e as regras de resposta.

Os chunks do PDF são persistidos durante a ingestão. Eles não são recriados a cada pergunta.

## 4. Estrutura do projeto

```text
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── document.pdf
└── README.md
```

Responsabilidades dos módulos:

- `src/ingest.py`: leitura, divisão, vetorização e persistência do PDF.
- `src/search.py`: recuperação semântica, construção do contexto e geração de respostas.
- `src/chat.py`: interface interativa de perguntas e respostas no terminal.

## 5. Pré-requisitos

Para executar o projeto, é necessário ter instalado:

- Python 3.11
- Docker com Docker Compose
- Git
- Uma chave de API válida do Google Gemini

A chave de API pode ser obtida no [Google AI Studio](https://aistudio.google.com/).

O uso da API pode gerar custos conforme os modelos utilizados e as condições da conta.

## 6. Instalação e configuração

### 6.1. Clonar o repositório

```bash
git clone https://github.com/Lesamsilva/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 6.2. Criar o ambiente virtual

No macOS ou Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 6.3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 6.4. Configurar as variáveis de ambiente

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Configure os seguintes valores:

```dotenv
GOOGLE_API_KEY=SUA_CHAVE_API
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GOOGLE_CHAT_MODEL=gemini-3.8-flash

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documentos_rag

PDF_PATH=document.pdf
```

Os valores de conexão devem corresponder à configuração do `docker-compose.yml`.

**Segurança:** o arquivo `.env` contém informações sensíveis e não deve ser enviado ao GitHub. Utilize o `.env.example` apenas como modelo de configuração, sem credenciais reais.

## 7. Execução da aplicação

Execute os comandos a partir da raiz do projeto, com o ambiente virtual ativado.

### 7.1. Inicializar o PostgreSQL

```bash
docker compose up -d
```

Verifique se o contêiner está em execução:

```bash
docker compose ps
```

A aplicação utiliza PostgreSQL com a extensão pgVector para armazenar os embeddings.

### 7.2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Esse processo:

1. Lê o arquivo `document.pdf`.
2. Divide o conteúdo em chunks.
3. Gera os embeddings com o Gemini.
4. Persiste os dados no PostgreSQL com pgVector.

**Importante:** a ingestão deve ser executada inicialmente para preparar a base de conhecimento. A implementação atual não possui uma estratégia explícita de deduplicação; repetir a ingestão na mesma coleção pode gerar registros duplicados.

### 7.3. Executar o chat

```bash
python src/chat.py
```

Após a inicialização, o programa permanece aguardando perguntas.

Para encerrar, digite `sair` ou utilize Ctrl+C.

## 8. Exemplo de utilização

```text
Chat RAG iniciado. Digite sua pergunta ou 'sair' para encerrar.

Você: Quais informações sobre as empresas estão disponíveis no documento?

Com base no documento, as informações disponíveis sobre as empresas são:

- Nome da empresa
- Faturamento
- Ano de fundação

Você: Qual é a capital da França?

Não tenho informações necessárias para responder sua pergunta.

Você: sair

Chat encerrado.
```

## 9. Regras para geração das respostas

O modelo recebe um prompt com o contexto recuperado e as seguintes restrições:

- Responder somente com base no contexto fornecido.
- Não utilizar conhecimento externo.
- Não inventar informações.
- Não produzir opiniões ou interpretações além do conteúdo disponível.
- Quando não houver informações suficientes, retornar:

```text
Não tenho informações necessárias para responder sua pergunta.
```

O prompt também inclui exemplos de perguntas fora do contexto para orientar o comportamento do modelo.

Essas instruções reduzem o risco de respostas não fundamentadas, mas não eliminam completamente a possibilidade de erros do modelo.

## 10. Repositório e entrega

Projeto desenvolvido a partir de um fork do repositório disponibilizado para o desafio:

[Repositório original](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca)

**Implementação:**

[Lesamsilva/mba-ia-desafio-ingestao-busca](https://github.com/Lesamsilva/mba-ia-desafio-ingestao-busca)

O repositório contém o código-fonte, a configuração do ambiente e as instruções necessárias para executar a aplicação.