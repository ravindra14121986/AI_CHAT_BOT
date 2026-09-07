"""Resolve a user request through the FAQ table and query ServiceNow."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin

import requests

from faq_logging import format_query_details, log_error, log_operation, log_success
from faq_query import connect_database, load_settings, search_faq


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_CONFIG = PROJECT_DIR / "secrent" / "db_config.cfg"
DEFAULT_SNOW_CONFIG = PROJECT_DIR / "secrent" / "snow.config"
INCIDENT_NUMBER_PATTERN = re.compile(r"\bINC\d{7}\b", re.IGNORECASE)
INCIDENT_PLACEHOLDER_PATTERN = re.compile(r"\bINCX{7}\b", re.IGNORECASE)


def normalize_incident_numbers(user_query: str) -> tuple[str, list[str]]:
    """Replace concrete incident numbers with the FAQ placeholder.

    Args:
        user_query: Original request, for example ``"Show INC1234567"``.

    Returns:
        A tuple containing the normalized lookup text and the original incident
        numbers in their appearance order.

    Example:
        ``normalize_incident_numbers("Show INC1234567")`` returns
        ``("Show INCXXXXXXX", ["INC1234567"])``.
    """
    incident_numbers = INCIDENT_NUMBER_PATTERN.findall(user_query)
    normalized_query = INCIDENT_NUMBER_PATTERN.sub("INCXXXXXXX", user_query)
    return normalized_query, incident_numbers


def restore_incident_numbers(api_filter: str, incident_numbers: list[str]) -> str:
    """Replace FAQ incident placeholders with values from the user request.

    Placeholder replacement follows appearance order. If no concrete incident
    number was supplied, the FAQ filter is returned unchanged.

    Args:
        api_filter: Filter returned by the FAQ row, such as
            ``"number=INCXXXXXXX^active=true"``.
        incident_numbers: Original incident numbers extracted from the request.

    Returns:
        The ServiceNow filter with placeholders restored.

    Example:
        ``restore_incident_numbers("number=INCXXXXXXX", ["INC1234567"])``
        returns ``"number=INC1234567"``.
    """
    numbers = iter(incident_numbers)

    def replace_placeholder(_: re.Match[str]) -> str:
        return next(numbers, "INCXXXXXXX")

    return INCIDENT_PLACEHOLDER_PATTERN.sub(replace_placeholder, api_filter)


def clean_value(value: str) -> str:
    """Remove optional matching quotes from a config value.

    Args:
        value: Raw INI value, such as ``'"https://instance.service-now.com"'``.

    Returns:
        The unquoted, trimmed value.

    Example:
        ``clean_value('"incident"')`` returns ``"incident"``.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def load_snow_settings(config_path: Path) -> dict[str, Any]:
    """Load ServiceNow connection and API settings from ``snow.config``.

    Args:
        config_path: Path to the ServiceNow configuration file.

    Returns:
        A dictionary containing the instance URL, credentials, table, and limits.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If required values are missing.

    Example:
        ``settings = load_snow_settings(Path("secrent/snow.config"))``.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"ServiceNow config not found: {config_path}")

    parser = configparser.RawConfigParser()
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("servicenow") or not parser.has_section("api"):
        raise ValueError("snow.config must contain [servicenow] and [api] sections")

    snow = parser["servicenow"]
    required = ("instance_url", "username", "password")
    missing = [key for key in required if not clean_value(snow.get(key, ""))]
    if missing:
        raise ValueError(f"Missing ServiceNow settings: {', '.join(missing)}")

    api = parser["api"]
    return {
        "instance_url": clean_value(snow["instance_url"]).rstrip("/") + "/",
        "username": clean_value(snow["username"]),
        "password": clean_value(snow["password"]),
        "timeout": int(clean_value(snow.get("timeout", "15"))),
        "table": clean_value(api.get("table", "incident")),
        "limit": int(clean_value(api.get("limit", "100"))),
    }


def build_servicenow_url(settings: dict[str, Any], api_filter: str, output_column: str) -> str:
    """Build the ServiceNow Table API URL from an FAQ result.

    Args:
        settings: ServiceNow settings from ``load_snow_settings``.
        api_filter: ServiceNow encoded query from the FAQ ``filter`` column.
        output_column: Comma-separated fields requested by the FAQ row.

    Returns:
        A URL containing ``sysparm_query`` and ``sysparm_fields`` parameters.

    Example:
        ``build_servicenow_url(settings, "priority=1", "number,priority")``.
    """
    from urllib.parse import urlencode

    fields = ",".join(field.strip() for field in output_column.split(",") if field.strip())
    if not fields:
        raise ValueError("FAQ output_column is empty")
    params = {
        "sysparm_query": api_filter,
        "sysparm_fields": fields,
        "sysparm_limit": str(settings["limit"]),
        "sysparm_display_value": "true",
    }
    return urljoin(settings["instance_url"], f"api/now/table/{settings['table']}") + "?" + urlencode(params)


def query_servicenow(
    settings: dict[str, Any],
    api_filter: str,
    output_column: str,
    connection: Any | None = None,
    user_name: str = "unknown",
    logging_table: str = "faq_agent_logging",
) -> list[dict[str, Any]]:
    """Call ServiceNow and return only fields listed in ``output_column``.

    Args:
        settings: ServiceNow connection and API settings.
        api_filter: Encoded ServiceNow query from the FAQ table.
        output_column: Comma-separated ServiceNow fields to return.
        connection: Optional PostgreSQL connection used for audit logging.
        user_name: Current chatbot user recorded in audit logs.
        logging_table: PostgreSQL audit table name.

    Returns:
        ServiceNow result records containing only requested fields.

    Raises:
        requests.RequestException: If the HTTP request fails.
        RuntimeError: If ServiceNow returns a non-success response.

    Example:
        ``query_servicenow(settings, "active=true", "number,state")``.
    """
    operation = "query_servicenow"
    try:
        url = build_servicenow_url(settings, api_filter, output_column)
        if connection is not None:
            log_operation(
                connection,
                user_name,
                format_query_details(operation, url, ("ServiceNow Basic Auth omitted",)),
                logging_table,
            )
        response = requests.get(
            url,
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=settings["timeout"],
        )
        if connection is not None:
            log_operation(connection, user_name, f"HTTP {operation}: status={response.status_code}", logging_table)
        if not response.ok:
            raise RuntimeError(f"ServiceNow returned HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        fields = [field.strip() for field in output_column.split(",") if field.strip()]
        rows = [{field: record.get(field) for field in fields} for record in payload.get("result", [])]
        if connection is not None:
            log_success(connection, user_name, operation, f"http_status={response.status_code}; rows={len(rows)}")
        return rows
    except Exception as error:
        if connection is not None:
            log_error(connection, user_name, operation, error)
        raise


def mutate_servicenow(
    settings: dict[str, Any],
    method: str,
    fields: dict[str, Any],
    identifier: str | None = None,
    connection: Any | None = None,
    user_name: str = "unknown",
    logging_table: str = "faq_agent_logging",
) -> dict[str, Any]:
    """Create or update a ServiceNow record through the Table API.

    ``POST`` creates a record from ``fields``. ``PATCH`` updates one record;
    incident numbers are resolved to ``sys_id`` before the update because the
    Table API record endpoint requires a sys_id path segment.
    """
    operation = "mutate_servicenow"
    normalized_method = method.upper()
    if normalized_method not in {"POST", "PATCH"}:
        raise ValueError(f"Unsupported mutation method: {method!r}")
    if not fields:
        raise ValueError("ServiceNow mutation fields cannot be empty")
    if normalized_method == "PATCH" and not identifier:
        raise ValueError("PATCH requires a ServiceNow record identifier")

    try:
        base_url = urljoin(settings["instance_url"], f"api/now/table/{settings['table']}")
        target_url = base_url
        if normalized_method == "PATCH":
            record_id = identifier
            if not re.fullmatch(r"[0-9a-fA-F]{32}", identifier or ""):
                lookup_params = urlencode(
                    {"sysparm_query": f"number={identifier}", "sysparm_fields": "sys_id", "sysparm_limit": "1"}
                )
                lookup_url = f"{base_url}?{lookup_params}"
                if connection is not None:
                    log_operation(connection, user_name, f"LOOKUP {operation}: identifier={identifier}", logging_table)
                lookup_response = requests.get(
                    lookup_url,
                    auth=(settings["username"], settings["password"]),
                    headers={"Accept": "application/json"},
                    timeout=settings["timeout"],
                )
                if not lookup_response.ok:
                    raise RuntimeError(
                        f"ServiceNow identifier lookup returned HTTP {lookup_response.status_code}: "
                        f"{lookup_response.text[:500]}"
                    )
                lookup_rows = lookup_response.json().get("result", [])
                if not lookup_rows:
                    raise LookupError(f"No ServiceNow record found for identifier {identifier}")
                record_id = lookup_rows[0].get("sys_id")
            if not record_id:
                raise LookupError(f"ServiceNow record has no sys_id for identifier {identifier}")
            target_url = f"{base_url}/{quote(record_id, safe='')}"

        if connection is not None:
            log_operation(connection, user_name, f"REQUEST {operation}: method={normalized_method}; fields={fields}", logging_table)
        response = requests.request(
            normalized_method,
            target_url,
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=fields,
            timeout=settings["timeout"],
        )
        if connection is not None:
            log_operation(connection, user_name, f"HTTP {operation}: status={response.status_code}", logging_table)
        if not response.ok:
            raise RuntimeError(f"ServiceNow returned HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        result = payload.get("result", payload)
        if connection is not None:
            log_success(connection, user_name, operation, f"method={normalized_method}; status={response.status_code}")
        return result
    except Exception as error:
        if connection is not None:
            log_error(connection, user_name, operation, error)
        raise


def print_output(rows: list[dict[str, Any]]) -> None:
    """Print only the requested ServiceNow output fields as JSON.

    Args:
        rows: Filtered ServiceNow result records.

    Example:
        ``print_output([{"number": "INC0012345", "priority": "1"}])``.
    """
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    """Parse the command-line request and configuration paths.

    Returns:
        Parsed arguments.

    Example:
        ``python faq_snow_query.py --query "Show high priority incidents"``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="User request to resolve through the FAQ table")
    parser.add_argument("--user-name", default="unknown", help="Chatbot user recorded in FAQ audit logs")
    parser.add_argument("--db-config", type=Path, default=DEFAULT_DB_CONFIG)
    parser.add_argument("--snow-config", type=Path, default=DEFAULT_SNOW_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Resolve the FAQ request, call ServiceNow, and print only requested fields.

    Returns:
        ``0`` on success and ``1`` on a handled failure.

    Example:
        ``python faq_snow_query.py --query "Show me high priority tickets"``.
    """
    args = parse_args()
    connection = None
    try:
        db_settings = load_settings(args.db_config)
        snow_settings = load_snow_settings(args.snow_config)
        connection = connect_database(db_settings)
        log_operation(connection, args.user_name, "START faq_snow_query.main", db_settings.logging_table)
        normalized_query, incident_numbers = normalize_incident_numbers(args.query)
        log_operation(
            connection,
            args.user_name,
            f"NORMALIZE faq_snow_query.main: query={normalized_query}; incident_count={len(incident_numbers)}",
            db_settings.logging_table,
        )
        faq_rows = search_faq(connection, db_settings, normalized_query, 1, args.user_name)
        if not faq_rows:
            raise LookupError("No FAQ mapping found for the user request")

        faq_row = faq_rows[0]
        api_filter = faq_row.get("api_query")
        output_column = faq_row.get("answer")
        if not api_filter or not output_column:
            raise ValueError("FAQ row must contain filter and output_column values")

        api_filter = restore_incident_numbers(api_filter, incident_numbers)
        log_operation(
            connection,
            args.user_name,
            f"RESTORE faq_snow_query.main: filter={api_filter}; output_column={output_column}",
            db_settings.logging_table,
        )
        rows = query_servicenow(
            snow_settings,
            api_filter,
            output_column,
            connection,
            args.user_name,
            db_settings.logging_table,
        )
        print_output(rows)
        log_success(connection, args.user_name, "faq_snow_query.main", f"rows={len(rows)}")
        return 0
    except Exception as error:
        if connection is not None:
            log_error(connection, getattr(args, "user_name", "unknown"), "faq_snow_query.main", error)
        print(f"ServiceNow query failed: {error}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
