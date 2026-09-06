"""Query the ServiceNow incident FAQ PostgreSQL table."""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from query_bulder import FAQColumns, build_faq_search_query, build_intent_query
from faq_logging import format_query_details, log_error, log_operation, log_success


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "secrent" / "db_config.cfg"


def clean_config_value(value: str) -> str:
    """Remove optional matching quotes from an INI value.

    Example:
        ``clean_config_value('"user_query"')`` returns ``"user_query"``.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


@dataclass(frozen=True)
class DatabaseSettings:
    """Database connection and FAQ schema settings.

    Example:
        ``settings = load_settings(Path("secrent/db_config.cfg"))``
        reads the connection values and configured table columns.
    """

    connection: dict[str, Any]
    table: str
    columns: FAQColumns
    logging_table: str = "faq_agent_logging"


def load_settings(config_path: Path) -> DatabaseSettings:
    """Load PostgreSQL and FAQ table settings from an INI config file.

    Args:
        config_path: Path to ``secrent/db_config.cfg``.

    Returns:
        A validated ``DatabaseSettings`` object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections or values are missing.

    Example:
        ``load_settings(Path("secrent/db_config.cfg"))`` loads the default
        database configuration.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Database config not found: {config_path}")

    parser = configparser.RawConfigParser()
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("postgresql") or not parser.has_section("faq"):
        raise ValueError("Config must contain [postgresql] and [faq] sections")

    database = parser["postgresql"]
    required = ("host", "port", "database", "user", "password")
    missing = [key for key in required if not database.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing database settings: {', '.join(missing)}")

    connection: dict[str, Any] = {
        "host": clean_config_value(database["host"]),
        "port": database.getint("port", fallback=5432),
        "dbname": clean_config_value(database["database"]),
        "user": clean_config_value(database["user"]),
        "password": clean_config_value(database["password"]),
        "sslmode": clean_config_value(database.get("sslmode", "prefer")),
        "connect_timeout": database.getint("connect_timeout", fallback=10),
    }
    faq = parser["faq"]
    columns = FAQColumns(
        id=clean_config_value(faq.get("id_column", "id")),
        question=clean_config_value(faq.get("question_column", "question")),
        answer=clean_config_value(faq.get("answer_column", "answer")),
        intent=clean_config_value(faq.get("intent_column", "intent")),
        api_query=clean_config_value(faq.get("api_query_column", "api_query")),
    )
    logging_section = parser["logging"] if parser.has_section("logging") else {}
    logging_table = clean_config_value(
        logging_section.get("table", "faq_agent_logging")
        if hasattr(logging_section, "get")
        else "faq_agent_logging"
    )
    return DatabaseSettings(
        connection,
        clean_config_value(faq.get("table", "servicenow_incident_faq")),
        columns,
        logging_table,
    )


def connect_database(settings: DatabaseSettings) -> Any:
    """Open a PostgreSQL connection using the configured settings.

    The import is delayed so query-builder tests can run without a database
    driver installed.

    Args:
        settings: Database connection settings returned by ``load_settings``.

    Returns:
        An open psycopg2 connection.

    Raises:
        RuntimeError: If ``psycopg2-binary`` is not installed.
        psycopg2.Error: If PostgreSQL rejects the connection.

    Example:
        ``connection = connect_database(settings)`` opens the database session.
    """
    try:
        import psycopg2
    except ImportError as error:
        raise RuntimeError("Install the PostgreSQL driver with: pip install psycopg2-binary") from error
    return psycopg2.connect(**settings.connection)


def search_faq(
    connection: Any,
    settings: DatabaseSettings,
    phrase: str,
    limit: int,
    user_name: str = "unknown",
) -> list[dict[str, Any]]:
    """Search FAQ questions, answers, and intents for a user phrase.

    Args:
        connection: Open PostgreSQL connection.
        settings: Table and column configuration.
        phrase: User search text, for example ``"VPN keeps disconnecting"``.
        limit: Maximum number of rows to return.

    Returns:
        Matching FAQ rows represented as dictionaries.

    Example:
        ``search_faq(connection, settings, "VPN timeout", 5)`` returns up to
        five matching FAQ records.
    """
    operation = "search_faq"
    log_operation(connection, user_name, f"START {operation}", settings.logging_table)
    try:
        if not phrase.strip():
            raise ValueError("Search phrase cannot be empty")
        sql = build_faq_search_query(settings.table, settings.columns, limit)
        parameters = (f"%{phrase.strip()}%",)
        log_operation(connection, user_name, format_query_details(operation, sql, parameters), settings.logging_table)
        rows = execute_query(connection, sql, parameters, settings.columns, user_name, settings.logging_table)
        log_success(connection, user_name, operation, f"rows={len(rows)}")
        return rows
    except Exception as error:
        log_error(connection, user_name, operation, error)
        raise


def search_by_intent(
    connection: Any,
    settings: DatabaseSettings,
    intent: str,
    limit: int,
    user_name: str = "unknown",
) -> list[dict[str, Any]]:
    """Fetch FAQ records for one intent label.

    Args:
        connection: Open PostgreSQL connection.
        settings: Table and column configuration.
        intent: Intent value, for example ``"get_incident_by_number"``.
        limit: Maximum number of rows to return.

    Returns:
        FAQ rows matching the exact intent.

    Example:
        ``search_by_intent(connection, settings, "count_incidents", 10)``
        returns FAQ records for that intent.
    """
    operation = "search_by_intent"
    log_operation(connection, user_name, f"START {operation}", settings.logging_table)
    try:
        if not intent.strip():
            raise ValueError("Intent cannot be empty")
        sql = build_intent_query(settings.table, settings.columns, limit)
        parameters = (intent.strip(),)
        log_operation(connection, user_name, format_query_details(operation, sql, parameters), settings.logging_table)
        rows = execute_query(connection, sql, parameters, settings.columns, user_name, settings.logging_table)
        log_success(connection, user_name, operation, f"rows={len(rows)}")
        return rows
    except Exception as error:
        log_error(connection, user_name, operation, error)
        raise


def execute_query(
    connection: Any,
    sql: str,
    parameters: tuple[Any, ...],
    columns: FAQColumns,
    user_name: str = "unknown",
    logging_table: str = "faq_agent_logging",
) -> list[dict[str, Any]]:
    """Execute parameterized SQL and map rows to dictionaries.

    Args:
        connection: Open PostgreSQL connection.
        sql: SQL containing PostgreSQL ``%s`` placeholders.
        parameters: Values bound to the SQL placeholders.
        columns: Column configuration used to label returned values.

    Returns:
        A list of dictionaries with FAQ fields.

    Example:
        ``execute_query(connection, sql, ("%VPN%",), settings.columns)``
        executes a safe parameterized search.
    """
    operation = "execute_query"
    log_operation(connection, user_name, format_query_details(operation, sql, parameters), logging_table)
    try:
        field_names = ["id", "question", "answer", "intent", "api_query"]
        with connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = [dict(zip(field_names, row)) for row in cursor.fetchall()]
        log_success(connection, user_name, operation, f"rows={len(rows)}")
        return rows
    except Exception as error:
        log_error(connection, user_name, operation, error)
        raise


def print_results(rows: list[dict[str, Any]]) -> None:
    """Print FAQ records in a readable format.

    Args:
        rows: Records returned by ``search_faq`` or ``search_by_intent``.

    Example:
        ``print_results(rows)`` displays each matching question and answer.
    """
    if not rows:
        print("No FAQ records found.")
        return
    for index, row in enumerate(rows, start=1):
        print(f"\n[{index}] Intent: {row.get('intent')}")
        print(f"Question: {row.get('question')}")
        print(f"output_column: {row.get('answer')}")
        print(f"API query: {row.get('api_query')}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for FAQ searches.

    Returns:
        Parsed command-line arguments.

    Example:
        ``python faq_query.py --query "VPN issue" --limit 5``
        searches the FAQ table.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--user-name",
        default=os.getenv("FAQ_AGENT_USER", "unknown"),
        help="Current chatbot user recorded in faq_agent_logging",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Search phrase for FAQ question/answer text")
    group.add_argument("--intent", help="Exact intent label to look up")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    """Load configuration, query PostgreSQL, and print FAQ results.

    Returns:
        Process exit code: ``0`` for success and ``1`` for a handled failure.

    Example:
        ``python faq_query.py --query "incident priority"`` runs a search.
    """
    args = parse_args()
    connection = None
    try:
        settings = load_settings(args.config)
        connection = connect_database(settings)
        log_operation(connection, args.user_name, "START main", settings.logging_table)
        if args.query is not None:
            rows = search_faq(connection, settings, args.query, args.limit, args.user_name)
        else:
            rows = search_by_intent(connection, settings, args.intent, args.limit, args.user_name)
        print_results(rows)
        log_success(connection, args.user_name, "main", f"rows={len(rows)}")
        return 0
    except Exception as error:
        if connection is not None:
            log_error(connection, getattr(args, "user_name", "unknown"), "main", error)
        print(f"Database query failed: {error}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
