"""Convert spaCy ServiceNow intent/entity predictions into API requests."""

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
DEFAULT_MODEL = PROJECT_DIR / "servicenow_model_run" / "output" / "model-best"
DEFAULT_DB_CONFIG = PROJECT_DIR / "secrent" / "db_config.cfg"
DEFAULT_SNOW_CONFIG = PROJECT_DIR / "secrent" / "snow.config"
DEFAULT_FIELDS = "number,short_description,priority,state,assigned_to,assignment_group,opened_at,resolved_at,closed_at"
LOGGER = logging.getLogger("spacy_servicenow_query")

QUERY_ENTITY_FIELDS = {
    "CATEGORY": "category",
    "LOCATION": "location",
    "ASSIGNMENT_GROUP": "assignment_group.name",
    "ASSIGNED_TO": "assigned_to.name",
    "CALLER": "caller_id.name",
    "OPENED_BY": "opened_by.name",
    "RESOLUTION_CODE": "close_code",
}
MUTATION_ENTITY_FIELDS = {
    "CATEGORY": "category",
    "LOCATION": "location",
    "ASSIGNMENT_GROUP": "assignment_group",
    "ASSIGNED_TO": "assigned_to",
    "CALLER": "caller_id",
    "OPENED_BY": "opened_by",
    "RESOLUTION_CODE": "close_code",
}
PRIORITY_VALUES = {"critical": "1", "p1": "1", "high": "2", "p2": "2", "medium": "3", "moderate": "3", "p3": "3", "low": "4", "p4": "4", "planning": "5", "p5": "5"}
URGENCY_VALUES = {"high": "1", "medium": "2", "low": "3"}
IMPACT_VALUES = {"high": "1", "medium": "2", "low": "3"}
STATE_VALUES = {
    "new": "1", "in progress": "2", "in-progress": "2", "on hold": "3", "on_hold": "3",
    "resolved": "6", "closed": "7", "canceled": "8", "cancelled": "8", "open": "1",
}
INCIDENT_PATTERN = re.compile(r"\bINC\d{7}\b", re.IGNORECASE)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_spacy_model(model_path: Path) -> Any:
    """Load the trained spaCy pipeline from disk."""
    LOGGER.info("START load_spacy_model: path=%s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"spaCy model not found: {model_path}")
    try:
        import spacy
        nlp = spacy.load(model_path)
    except ImportError as error:
        raise RuntimeError("Install spaCy with: pip install spacy") from error
    LOGGER.info("SUCCESS load_spacy_model: pipeline=%s", nlp.pipe_names)
    return nlp


def predict(nlp: Any, user_prompt: str) -> tuple[str, list[dict[str, str]]]:
    """Run text classification and NER and return the winning intent/entities."""
    if not user_prompt.strip():
        raise ValueError("User request cannot be empty")
    LOGGER.info("START predict: characters=%d", len(user_prompt))
    doc = nlp(user_prompt.strip())
    if not doc.cats:
        raise ValueError("spaCy model returned no intent scores")
    intent, score = max(doc.cats.items(), key=lambda item: item[1])
    entities = [{"label": entity.label_, "value": entity.text} for entity in doc.ents]
    LOGGER.info("SUCCESS predict: intent=%s score=%.4f entities=%s", intent, score, entities)
    return intent, entities


def normalize_value(label: str, value: str) -> str:
    """Normalize human labels to ServiceNow choice values where applicable."""
    text = value.strip()
    lowered = text.lower().replace(" - ", " ")
    if label == "PRIORITY":
        return PRIORITY_VALUES.get(lowered, text.split()[0] if text[:1].isdigit() else text)
    if label == "URGENCY":
        return URGENCY_VALUES.get(lowered, text.split()[0] if text[:1].isdigit() else text)
    if label == "IMPACT":
        return IMPACT_VALUES.get(lowered, text.split()[0] if text[:1].isdigit() else text)
    if label == "STATE":
        return STATE_VALUES.get(lowered, text.split()[0] if text[:1].isdigit() else text)
    return text


def build_request(user_prompt: str, intent: str, entities: list[dict[str, str]]) -> dict[str, Any]:
    """Build a ServiceNow request from spaCy labels and the original text."""
    operation = "build_request"
    extracted: dict[str, list[str]] = {}
    for entity in entities:
        extracted.setdefault(entity["label"], []).append(entity["value"])

    incident_number = next(iter(extracted.get("INCIDENT_NUMBER", [])), None)
    method = "GET" if intent in {"GET_INCIDENT", "GET_OPENED_BY", "LIST_INCIDENTS_BY_STATE", "LIST_INCIDENTS_BY_CALLER", "SEARCH_INCIDENT_BY_KEYWORD"} else "PATCH"
    if intent == "CREATE_INCIDENT":
        method = "POST"

    fields: dict[str, Any] = {}
    query_parts: list[str] = []
    for label, values in extracted.items():
        if label == "INCIDENT_NUMBER":
            if method == "GET":
                query_parts.append(f"number={values[0].upper()}")
        elif label in {"PRIORITY", "URGENCY", "IMPACT", "STATE"}:
            field = label.lower()
            value = normalize_value(label, values[-1])
            if method == "GET":
                query_parts.append(f"{field}={value}")
            else:
                fields[field] = value
        elif label in QUERY_ENTITY_FIELDS:
            query_field = QUERY_ENTITY_FIELDS[label]
            mutation_field = MUTATION_ENTITY_FIELDS[label]
            if method == "GET":
                query_parts.append(f"{query_field}={values[-1]}")
            else:
                fields[mutation_field] = values[-1]

    if intent == "REOPEN_INCIDENT":
        fields["state"] = fields.get("state", STATE_VALUES["in progress"])
    if intent == "CLOSE_INCIDENT":
        fields["state"] = fields.get("state", STATE_VALUES["closed"])
    if intent == "SEARCH_INCIDENT_BY_KEYWORD" and not query_parts:
        query_parts.append(f"short_descriptionLIKE{user_prompt.strip()}")

    if method == "POST":
        description = next(iter(extracted.get("DESCRIPTION", [])), None)
        if description:
            fields["short_description"] = description
        else:
            fields["short_description"] = user_prompt.strip()
    missing: list[str] = []
    if method == "PATCH" and not incident_number:
        missing.append("incident number")
    if method == "POST" and not extracted.get("CALLER") and not extracted.get("OPENED_BY"):
        missing.append("caller")
    if intent == "UNKNOWN":
        method, query_parts, fields = "", [], {}
    request = {
        "intent": intent,
        "table": "incident",
        "method": method,
        "query": "^".join(query_parts),
        "fields": fields,
        "identifier": incident_number.upper() if incident_number else None,
        "missing_info": missing,
        "clarification_needed": bool(missing),
        "entities": extracted,
    }
    LOGGER.info("SUCCESS %s: request=%s", operation, request)
    return request


def audit(connection: Any, user_name: str, details: str, table: str) -> None:
    if connection is not None:
        log_operation(connection, user_name, details, table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="User request; if omitted, read from stdin")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--execute", action="store_true", help="Send the generated request to ServiceNow")
    parser.add_argument("--user-name", default="unknown")
    parser.add_argument("--db-config", type=Path, default=DEFAULT_DB_CONFIG)
    parser.add_argument("--snow-config", type=Path, default=DEFAULT_SNOW_CONFIG)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    connection = None
    logging_table = "faq_agent_logging"
    try:
        user_prompt = args.query or input("ServiceNow request: ")
        db_settings = load_settings(args.db_config)
        logging_table = db_settings.logging_table
        connection = connect_database(db_settings)
        audit(connection, args.user_name, "START spacy_servicenow_query.main", logging_table)
        audit(connection, args.user_name, f"INPUT spacy_servicenow_query.main: characters={len(user_prompt)}", logging_table)
        nlp = load_spacy_model(args.model_path)
        intent, entities = predict(nlp, user_prompt)
        audit(connection, args.user_name, f"PREDICTION spacy_servicenow_query.main: intent={intent}; entities={entities}", logging_table)
        request = build_request(user_prompt, intent, entities)
        if request["clarification_needed"]:
            raise ValueError(f"Clarification required: {', '.join(request['missing_info'])}")
        result: dict[str, Any] = request.copy()
        if args.execute:
            snow_settings = {**load_snow_settings(args.snow_config), "table": request["table"]}
            if request["method"] == "GET":
                result["results"] = query_servicenow(snow_settings, request["query"], args.fields, connection, args.user_name, logging_table)
            else:
                result["results"] = mutate_servicenow(snow_settings, request["method"], request["fields"], request["identifier"], connection, args.user_name, logging_table)
            audit(connection, args.user_name, "SUCCESS spacy_servicenow_query.main", logging_table)
        else:
            LOGGER.info("SUCCESS main: request generated; execution skipped")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        if connection is not None:
            log_error(connection, args.user_name, "spacy_servicenow_query.main", error)
        LOGGER.error("spaCy ServiceNow request failed: %s", error)
        print(f"spaCy ServiceNow request failed: {error}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())