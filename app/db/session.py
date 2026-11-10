import os
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        if url.startswith("sqlite:///"):
            path = url.removeprefix("sqlite:///")
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
        _engine = create_engine(url, connect_args={"check_same_thread": False})
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    add_missing_columns(engine)


def add_missing_columns(engine) -> None:
    """Tiny forward-only migration: create_all() never alters existing tables, so add
    any *nullable* model column an older database lacks. Anything more complex
    (renames, NOT NULL, type changes) needs a real migration tool."""
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                if not col.nullable:
                    raise RuntimeError(
                        f"cannot auto-add NOT NULL column {table.name}.{col.name}; "
                        "delete the database or migrate it manually"
                    )
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


@contextmanager
def session_scope():
    get_engine()
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
