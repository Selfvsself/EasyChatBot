from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # JWT
    SECRET_KEY: str = "SUPER_SECRET_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 1

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_REQUEST_TOPIC: str = "chat_requests"
    KAFKA_RESPONSE_TOPIC: str = "chat_responses"

    # App
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8001

    # DATABASE
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/mydb"

    # LLM
    LLM_URL: str = "localhost:11434"
    LLM_MODEL: str = "gpt"
    RAG_ENABLED: bool = True
    RAG_EMBEDDING_URL: str = "localhost:11434/api/embed"
    RAG_EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    RAG_CHUNK_SIZE: int = 3500
    RAG_CHUNK_OVERLAP: int = 500
    RAG_TOP_K_CHUNKS_PER_RESULT: int = 1

    # Chat memory
    CHAT_MAX_ACTIVE_MESSAGES: int = 10
    CHAT_ARCHIVE_BATCH_SIZE: int = 5
    CHAT_MEMORY_MAX_CHARS: int = 5000

    # Jira
    JIRA_BASE_URL: str = ""
    JIRA_EMAIL: str = ""
    JIRA_API_TOKEN: str = ""
    JIRA_PAT: str = ""
    JIRA_PROJECT_KEY: str = ""

    # Confluence
    CONFLUENCE_BASE_URL: str = ""
    CONFLUENCE_EMAIL: str = ""
    CONFLUENCE_API_TOKEN: str = ""
    CONFLUENCE_PAT: str = ""
    CONFLUENCE_SPACE_KEY: str = ""

    class Config:
        env_file = ".env",
        env_file_encoding = "utf-8"


settings = Settings()
