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
