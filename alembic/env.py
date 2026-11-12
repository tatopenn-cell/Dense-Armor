"""Alembic environment: read the URL from DENSE_ARMOR_PG_URL or -x url=..."""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _url() -> str:
    if "url" in context.get_x_argument(as_dictionary=True):
        return context.get_x_argument(as_dictionary=True)["url"]
    env_url = os.environ.get("DENSE_ARMOR_PG_URL")
    if not env_url:
        raise RuntimeError(
            "set DENSE_ARMOR_PG_URL or pass -x url=<postgresql url> to alembic"
        )
    return env_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        cfg, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
