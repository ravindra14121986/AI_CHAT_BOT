"""Generate and optionally execute a ServiceNow request with a local Ollama model.

The model is constrained by ``servicenow_incident_system_prompt.txt`` and must
return the JSON contract documented in that file. GET, POST, and PATCH
execution is supported through the existing ServiceNow configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from faq_logging import log_error, log_operation
from faq_query import connect_database, load_settings
from faq_snow_query import load_snow_settings, mutate_servicenow, query_servicenow


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = PROJECT_DIR / "servicenow_incident_system_prompt.txt"
DEFAULT_DB_CONFIG = PROJECT_DIR / "secrent" / "db_config.cfg"
DEFAULT_SNOW_CONFIG = PROJECT_DIR / "secrent" / "snow.config"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_FIELDS = "number,short_description,priority,state,assigned_to,assignment_group,opened_at,resolved_at,closed_at"
LOGGER = logging.getLogger("ollama_servicenow_query")
REQUIRED_RESPONSE_KEYS = {
    "intent",
    "table",
    "method",
    "query",
    "fields",
    "identifier",
    "missing_info",
    "clarification_needed",
    "raw_entities",
}


def configure_logging() -> None:
    """Configure concise console logging for each processing step."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_system_prompt(prompt_path: Path) -> str:
    """Read and validate the system prompt used for query generation."""
    operation = "read_system_prompt"
    LOGGER.info("START %s: path=%s", operation, prompt_path)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt is empty: {prompt_path}")
    LOGGER.info("SUCCESS %s: characters=%d", operation, len(prompt))
    return prompt


def extract_json(content: str) -> dict[str, Any]:
    """Parse the model response, tolerating accidental markdown fences."""
    operation = "parse_model_response"
    LOGGER.info("START %s: response_characters=%d", operation, len(content))
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"Ollama returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Ollama response must be a JSON object")
    missing = REQUIRED_RESPONSE_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Ollama response is missing keys: {', '.join(sorted(missing))}")
    if not isinstance(payload["query"], str) or not isinstance(payload["fields"], dict):
        raise ValueError("Ollama response fields 'query' and 'fields' have invalid types")
    LOGGER.info(
        "SUCCESS %s: intent=%s method=%s query=%s",
        operation,
        payload["intent"],
        payload["method"],
        payload["query"] or "<empty>",
    )
    return payload


def generate_query(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    """Ask Ollama to classify the request and generate its ServiceNow query."""
    operation = "generate_query"
    if not user_prompt.strip():
        raise ValueError("User prompt cannot be empty")
    LOGGER.info("START %s: model=%s user_prompt_characters=%d", operation, model, len(user_prompt))
    try:
        import ollama

        response = ollama.chat(
            model=model,
            think=False,
            format="json",
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )
    except ImportError as error:
        raise RuntimeError("Install the Ollama client with: pip install ollama") from error
    except Exception as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error

    content = response.message.content
    return extract_json(content)


def validate_request(payload: dict[str, Any]) -> None:
    """Ensure the generated operation can be passed to the ServiceNow API."""
    operation = "validate_request"
    method = str(payload.get("method", "")).upper()
    intent = payload.get("intent")
    if payload.get("clarification_needed"):
        missing = payload.get("missing_info") or ["additional information"]
        raise ValueError(f"Clarification required for {intent}: {', '.join(map(str, missing))}")
    if method not in {"GET", "POST", "PATCH"}:
        raise ValueError(f"Generated method {method!r} is not supported")
    if not payload.get("table"):
        raise ValueError("Generated ServiceNow table is empty")
    if method == "GET" and not payload.get("query"):
        raise ValueError("Generated ServiceNow query is empty")
    if method == "POST" and not payload.get("fields"):
        raise ValueError("Generated create fields are empty")
    if method == "PATCH" and not payload.get("identifier"):
        raise ValueError("Generated update identifier is empty")
    LOGGER.info("SUCCESS %s: intent=%s table=%s", operation, intent, payload["table"])


def audit(connection: Any | None, user_name: str, details: str, logging_table: str) -> None:
    """Write an audit event when a database connection is available."""
    if connection is not None:
        log_operation(connection, user_name, details, logging_table)


def parse_args() -> argparse.Namespace:
    """Parse the user request and optional execution settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="User request; if omitted, read one line from stdin")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--fields", default=DEFAULT_FIELDS, help="Comma-separated fields returned by ServiceNow")
    parser.add_argument("--execute", action="store_true", help="Execute the generated ServiceNow request")
    parser.add_argument("--user-name", default="unknown")
    parser.add_argument(
        "--db-config",
        type=Path,
        default=DEFAULT_DB_CONFIG,
        help=f"PostgreSQL config file used for the existing audit connection (default: {DEFAULT_DB_CONFIG})",
    )
    parser.add_argument("--snow-config", type=Path, default=DEFAULT_SNOW_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Generate a query, print it, and optionally retrieve ServiceNow results."""
    configure_logging()
    args = parse_args()
    connection = None
    logging_table = "faq_agent_logging"
    try:
        user_prompt = args.query or input("ServiceNow request: ")
        system_prompt = read_system_prompt(args.system_prompt)

        LOGGER.info("START database_connection: config=%s", args.db_config)
        db_settings = load_settings(args.db_config)
        logging_table = db_settings.logging_table
        connection = connect_database(db_settings)
        LOGGER.info("SUCCESS database_connection: logging_table=%s", logging_table)
        audit(connection, args.user_name, "START ollama_servicenow_query.main", logging_table)
        audit(connection, args.user_name, f"INPUT ollama_servicenow_query.main: characters={len(user_prompt)}", logging_table)

        payload = generate_query(system_prompt, user_prompt, args.model)
        audit(
            connection,
            args.user_name,
            f"MODEL ollama_servicenow_query.main: intent={payload['intent']}; method={payload['method']}",
            logging_table,
        )
        validate_request(payload)
        audit(connection, args.user_name, f"QUERY ollama_servicenow_query.main: sysparm_query={payload['query']}", logging_table)

        result: dict[str, Any] = {
            "intent": payload["intent"],
            "table": payload["table"],
            "method": payload["method"],
            "sysparm_query": payload["query"],
            "sysparm_fields": args.fields,
            "fields": payload["fields"],
            "identifier": payload["identifier"],
            "entities": payload["raw_entities"],
        }
        if args.execute:
            snow_settings = load_snow_settings(args.snow_config)
            request_settings = {**snow_settings, "table": payload["table"]}
            if payload["method"].upper() == "GET":
                response_data: Any = query_servicenow(
                    request_settings,
                    payload["query"],
                    args.fields,
                    connection,
                    args.user_name,
                    logging_table,
                )
            else:
                response_data = mutate_servicenow(
                    request_settings,
                    payload["method"],
                    payload["fields"],
                    payload["identifier"],
                    connection,
                    args.user_name,
                    logging_table,
                )
            result["results"] = response_data
            log_operation(connection, args.user_name, "SUCCESS ollama_servicenow_query.main", logging_table)
        else:
            LOGGER.info("SUCCESS main: query generated; execution skipped")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        if connection is not None:
            log_error(connection, args.user_name, "ollama_servicenow_query.main", error)
        LOGGER.error("ServiceNow query generation failed: %s", error)
        print(f"ServiceNow query generation failed: {error}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())