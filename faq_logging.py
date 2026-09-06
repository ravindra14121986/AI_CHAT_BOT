"""Database-backed audit logging for the ServiceNow FAQ agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from query_bulder import quote_identifier


DEFAULT_LOG_TABLE = "faq_agent_logging"


def utc_timestamp() -> str:
    """Return the current UTC timestamp as an ISO-8601 string.

    Returns:
        A UTC timestamp such as ``"2026-09-06T14:32:10.123456+00:00"``.

    Example:
        ``timestamp = utc_timestamp()`` creates a value for ``create_timestamp``.
    """
    return datetime.now(timezone.utc).isoformat()


def log_operation(
    connection: Any,
    user_name: str,
    operation_details: str,
    table: str = DEFAULT_LOG_TABLE,
) -> bool:
    """Insert one audit event into the FAQ logging table.

    Logging is deliberately best-effort: an audit-table failure is rolled back
    and reported to stderr, but it does not hide or replace the FAQ operation's
    original result or exception.

    Args:
        connection: Open PostgreSQL connection.
        user_name: Current chatbot user, for example ``"alice"``.
        operation_details: Safe operation description. Do not include database
            passwords or other credentials.
        table: Logging table name, normally ``"faq_agent_logging"``.

    Returns:
        ``True`` when the event is inserted, otherwise ``False``.

    Example:
        ``log_operation(connection, "alice", "search_faq started")`` records
        the event in ``faq_agent_logging``.
    """
    sql = (
        f"INSERT INTO {quote_identifier(table)} "
        f"({quote_identifier('operation_details')}, "
        f"{quote_identifier('create_timestamp')}, "
        f"{quote_identifier('user_name')}) VALUES (%s, %s, %s)"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (operation_details[:4000], utc_timestamp(), user_name[:255]))
        connection.commit()
        return True
    except Exception as error:
        connection.rollback()
        print(f"Warning: audit logging failed: {error}")
        return False



def format_query_details(function_name: str, sql: str, parameters: tuple[Any, ...]) -> str:
    """Format a generated SQL statement and its bound values for audit logs.

    SQL is normalized to one line so the log table remains easy to search.
    Parameters are recorded separately from the SQL template, which documents
    the dynamic query without interpolating values into executable SQL.

    Args:
        function_name: Function that built or executed the query.
        sql: SQL template containing ``%s`` placeholders.
        parameters: Values bound to those placeholders.

    Returns:
        A bounded log message containing the function name, SQL, and parameters.

    Example:
        ``format_query_details("search_faq", sql, ("%VPN%",))``.
    """
    normalized_sql = " ".join(sql.split())
    return f"QUERY {function_name}: sql={normalized_sql}; parameters={parameters!r}"


def log_success(connection: Any, user_name: str, function_name: str, details: str = "") -> bool:
    """Record a successful function completion.

    Args:
        connection: Open PostgreSQL connection.
        user_name: Current chatbot user.
        function_name: Function that completed, such as ``"search_faq"``.
        details: Optional non-sensitive summary, such as ``"rows=3"``.

    Returns:
        ``True`` when the event is written.

    Example:
        ``log_success(connection, "alice", "search_faq", "rows=3")``.
    """
    suffix = f": {details}" if details else ""
    return log_operation(connection, user_name, f"SUCCESS {function_name}{suffix}")


def log_error(connection: Any, user_name: str, function_name: str, error: Exception) -> bool:
    """Record a handled function error without recording secrets.

    Args:
        connection: Open PostgreSQL connection.
        user_name: Current chatbot user.
        function_name: Function where the error occurred.
        error: Exception whose type and sanitized message are recorded.

    Returns:
        ``True`` when the event is written.

    Example:
        ``log_error(connection, "alice", "execute_query", error)`` records
        the exception type and message.
    """
    return log_operation(connection, user_name, f"ERROR {function_name}: {type(error).__name__}: {error}")
