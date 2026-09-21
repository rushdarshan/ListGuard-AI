from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./listguard.db"
    redis_url: str = ""  # empty = skip redis in /ready; compose sets redis://redis:6379/0

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
