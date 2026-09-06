"""Safe, reusable SQL builders for the ServiceNow FAQ table."""

from __future__ import annotations

import re
from dataclasses import dataclass


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class FAQColumns:
    """Column names used by the FAQ table.

    Example:
        ``FAQColumns(question="user_question", answer="resolution")``
        adapts the builder to a table with different column names.
    """

    id: str = "id"
    question: str = "question"
    answer: str = "answer"
    intent: str = "intent"
    api_query: str = "api_query"


class QueryBuilderError(ValueError):
    """Raised when a query builder receives an invalid identifier or limit."""


def split_identifiers(value: str) -> list[str]:
    """Split one configured identifier or a comma-separated identifier list.

    Example:
        ``split_identifiers("number,short_description")`` returns two names.
    """
    identifiers = [part.strip() for part in value.split(",") if part.strip()]
    if not identifiers:
        raise QueryBuilderError("At least one SQL identifier is required")
    return identifiers


def quote_identifier(identifier: str) -> str:
    """Validate and quote a PostgreSQL identifier.

    Values such as search text must be query parameters, but table and column
    names cannot be parameters. This function accepts simple identifiers and
    quotes them safely for SQL.

    Args:
        identifier: A table or column name, for example ``"question"``.

    Returns:
        A double-quoted PostgreSQL identifier, for example ``"question"``.

    Raises:
        QueryBuilderError: If the identifier contains unsafe characters.

    Example:
        ``quote_identifier("servicenow_incident_faq")`` returns
        ``'"servicenow_incident_faq"'``.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise QueryBuilderError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def validate_limit(limit: int) -> int:
    """Validate the maximum number of rows returned by a query.

    Args:
        limit: Positive row limit, for example ``10``.

    Returns:
        The validated limit.

    Raises:
        QueryBuilderError: If the limit is not between 1 and 100.

    Example:
        ``validate_limit(25)`` returns ``25``; ``validate_limit(0)`` fails.
    """
    if not 1 <= limit <= 100:
        raise QueryBuilderError("limit must be between 1 and 100")
    return limit


def build_faq_search_query(
    table: str = "servicenow_incident_faq",
    columns: FAQColumns | None = None,
    limit: int = 5,
) -> str:
    """Build a parameterized FAQ search query filtered only by user query text.

    The returned SQL expects one parameter: the search phrase. PostgreSQL
    ``ILIKE`` searches only the configured question column, which is
    ``user_query`` in the application config.

    Args:
        table: FAQ table name, for example ``"servicenow_incident_faq"``.
        columns: Configured FAQ column names, or defaults when omitted.
        limit: Maximum rows to return, from 1 through 100.

    Returns:
        SQL containing a single ``%s`` placeholder for the search phrase.

    Example:
        ``build_faq_search_query(limit=10)`` can be executed with
        ``cursor.execute(sql, ("vpn timeout",))``.
    """
    columns = columns or FAQColumns()
    validate_limit(limit)
    answer_columns = split_identifiers(columns.answer)
    selected_answer = (
        quote_identifier(answer_columns[0])
        if len(answer_columns) == 1
        else "jsonb_build_object(" + ", ".join(
            f"'{column}', {quote_identifier(column)}" for column in answer_columns
        ) + ") AS \"answer\""
    )
    selected = ", ".join(
        [quote_identifier(columns.id), quote_identifier(columns.question), selected_answer,
         quote_identifier(columns.intent), quote_identifier(columns.api_query)]
    )
    table_name = quote_identifier(table)
    search_column = f"COALESCE({quote_identifier(columns.question)}, '')"
    return (
        f"SELECT {selected} FROM {table_name} "
        f"WHERE {search_column} ILIKE %s "
        f"ORDER BY {quote_identifier(columns.id)} DESC LIMIT {limit}"
    )


def build_intent_query(
    table: str = "servicenow_incident_faq",
    columns: FAQColumns | None = None,
    limit: int = 10,
) -> str:
    """Build a query for FAQs belonging to one intent.

    Args:
        table: FAQ table name.
        columns: Configured FAQ column names.
        limit: Maximum rows to return, from 1 through 100.

    Returns:
        SQL with one ``%s`` placeholder for the intent value.

    Example:
        ``build_intent_query()`` can be executed with
        ``cursor.execute(sql, ("get_incident_by_number",))``.
    """
    columns = columns or FAQColumns()
    validate_limit(limit)
    answer_columns = split_identifiers(columns.answer)
    selected_answer = (
        quote_identifier(answer_columns[0])
        if len(answer_columns) == 1
        else "jsonb_build_object(" + ", ".join(
            f"'{column}', {quote_identifier(column)}" for column in answer_columns
        ) + ") AS \"answer\""
    )
    selected = ", ".join(
        [quote_identifier(columns.id), quote_identifier(columns.question), selected_answer,
         quote_identifier(columns.intent), quote_identifier(columns.api_query)]
    )
    return (
        f"SELECT {selected} FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(columns.intent)} = %s "
        f"ORDER BY {quote_identifier(columns.id)} DESC LIMIT {limit}"
    )
