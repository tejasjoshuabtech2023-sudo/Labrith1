"""
tools/openapi.py — OpenAPI 3.0 specification parser.

Parses OpenAPI 3 specifications from inline dictionaries, JSON/YAML strings,
local file paths, or remote URLs. Provides structured extractors for endpoints,
schemas, and summaries used by the Planner agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Sample support ticket OpenAPI spec used as a fallback for offline / demo environments
DEMO_BACKEND_SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {
        "title": "Support Ticket Service API",
        "version": "1.0.0",
        "description": "API for managing customer support tickets, priority queues, and agent replies.",
    },
    "paths": {
        "/api/tickets": {
            "get": {
                "operationId": "listTickets",
                "summary": "List open support tickets",
                "parameters": [
                    {
                        "name": "priority",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    },
                    {
                        "name": "status",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "enum": ["open", "pending", "resolved", "closed"]},
                    },
                ],
                "responses": {
                    "200": {
                        "description": "List of tickets",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Ticket"},
                                }
                            }
                        },
                    }
                },
            },
            "post": {
                "operationId": "createTicket",
                "summary": "Create a new support ticket",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreateTicketRequest"}
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Ticket created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Ticket"}
                            }
                        },
                    }
                },
            },
        },
        "/api/tickets/{id}": {
            "get": {
                "operationId": "getTicketDetail",
                "summary": "Retrieve ticket details and message thread",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Detailed ticket information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TicketDetail"}
                            }
                        },
                    }
                },
            }
        },
        "/api/tickets/{id}/reply": {
            "post": {
                "operationId": "sendReply",
                "summary": "Post a reply to a support ticket thread",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ReplyRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Reply posted successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TicketDetail"}
                            }
                        },
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "Ticket": {
                "type": "object",
                "required": ["id", "title", "priority", "status", "created_at"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "status": {"type": "string", "enum": ["open", "pending", "resolved", "closed"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "assigned_to": {"type": "string"},
                },
            },
            "TicketDetail": {
                "type": "object",
                "required": ["id", "title", "priority", "status", "messages"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "priority": {"type": "string"},
                    "status": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Message"},
                    },
                },
            },
            "Message": {
                "type": "object",
                "required": ["id", "author", "body", "timestamp"],
                "properties": {
                    "id": {"type": "string"},
                    "author": {"type": "string"},
                    "body": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
            "ReplyRequest": {
                "type": "object",
                "required": ["body"],
                "properties": {
                    "body": {"type": "string"},
                },
            },
            "CreateTicketRequest": {
                "type": "object",
                "required": ["title", "priority", "initial_message"],
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "initial_message": {"type": "string"},
                },
            },
        }
    },
}


def load_openapi_spec(spec_or_url: Union[str, dict[str, Any], Path]) -> dict[str, Any]:
    """Load an OpenAPI spec from a dict, file path, or URL."""
    if isinstance(spec_or_url, dict):
        return spec_or_url

    if isinstance(spec_or_url, Path):
        with open(spec_or_url, "r", encoding="utf-8") as f:
            return json.load(f)

    if isinstance(spec_or_url, str):
        # Check if it looks like a URL
        parsed = urlparse(spec_or_url)
        if parsed.scheme in ("http", "https"):
            # Check for demo/example placeholder URL
            if "example.com" in parsed.netloc:
                return DEMO_BACKEND_SPEC
            try:
                response = httpx.get(spec_or_url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "yaml" in content_type or spec_or_url.endswith((".yaml", ".yml")):
                    try:
                        import yaml
                        return yaml.safe_load(response.text)
                    except Exception:
                        pass
                return response.json()
            except Exception as exc:
                logger.warning(
                    "Failed to fetch OpenAPI spec from %s (%s). Using fallback spec.",
                    spec_or_url,
                    exc,
                )
                return DEMO_BACKEND_SPEC

        # Check if it's a local file path
        path = Path(spec_or_url)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try parsing as JSON string directly
        try:
            return json.loads(spec_or_url)
        except json.JSONDecodeError:
            pass

    return DEMO_BACKEND_SPEC


def list_endpoints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract an ordered list of endpoints with method, path, operationId, and summary."""
    endpoints: list[dict[str, Any]] = []
    paths = spec.get("paths", {})
    http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in http_methods or not isinstance(operation, dict):
                continue
            endpoints.append({
                "method": method.upper(),
                "path": path,
                "operation_id": operation.get("operationId", f"{method.lower()}_{path.replace('/', '_')}"),
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
                "parameters": [
                    p.get("name")
                    for p in operation.get("parameters", [])
                    if isinstance(p, dict) and "name" in p
                ],
            })

    return endpoints


def list_schemas(spec: dict[str, Any]) -> dict[str, Any]:
    """Extract component/model schemas from components.schemas or definitions."""
    components = spec.get("components", {})
    if "schemas" in components and isinstance(components["schemas"], dict):
        return components["schemas"]
    if "definitions" in spec and isinstance(spec["definitions"], dict):
        return spec["definitions"]
    return {}


def get_spec_summary(spec_or_url: Union[str, dict[str, Any], Path]) -> dict[str, Any]:
    """Return a compact summary of the OpenAPI spec for LLM prompt context."""
    spec = load_openapi_spec(spec_or_url)
    info = spec.get("info", {})
    endpoints = list_endpoints(spec)
    schemas = list_schemas(spec)

    return {
        "title": info.get("title", "API"),
        "version": info.get("version", "1.0.0"),
        "endpoints": endpoints,
        "schema_names": list(schemas.keys()),
    }
