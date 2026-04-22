from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


def init_db():
    import models.app  # noqa: F401
    import models.chat  # noqa: F401
    import models.message  # noqa: F401
    import models.user  # noqa: F401
    import models.user_app  # noqa: F401
    import models.tool  # noqa: F401
    import models.app_tool  # noqa: F401

    Base.metadata.create_all(bind=engine)
