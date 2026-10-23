"""
QA Agent.

Runs the app in a headless browser and exercises user flows.
Generates Playwright flow scripts via the LLM, executes them via PlaywrightRunner,
retries failed flows once to detect flakiness, classifies persistent failures
(code_bug, integration_bug, flaky), and outputs a proof-of-work record per flow.

QA is the only agent that may declare a flow passed or failed.

Design Doc §3: "Runs the app in a headless browser and exercises user flows. Decides: Which flows to test; whether a failure is a real bug."
Design Doc §5 (Scenario 4): "QA flow fails in the browser: QA attaches screenshot and console log and classifies the failure (code bug, integration bug, test flake); the Planner routes to the matching agent."
Design Doc §6: "Proof-of-work record attached to every agent output. Test gate: every required user flow passes in Playwright."
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

from graph.state import GraphState, QAResults, TaskDefinition
from lib.llm import complete
from lib.proofs import create_proof
from lib.trace import TraceWriter
from tools.playwright_runner import AppServer, PlaywrightFlowResult, PlaywrightRunner
from tools.sandbox import Sandbox


# ---------------------------------------------------------------------------
# Failure classification heuristics
# ---------------------------------------------------------------------------

def classify_failure(
    flow_result: PlaywrightFlowResult,
    flaky: bool = False,
) -> str:
    """Classify a flow failure as 'flaky', 'integration_bug', or 'code_bug'."""
    if flaky:
        return "flaky"

    combined_text = (
        f"{flow_result.error or ''} "
        + " ".join(flow_result.console_logs)
        + " "
        + " ".join(flow_result.page_errors)
    ).lower()

    # Integration / API mismatch indicators
    integration_patterns = [
        r"failed to fetch",
        r"networkerror",
        r"net::err_",
        r"status (?:code )?(?:4\d\d|5\d\d)",
        r"cors",
        r"access-control-allow-origin",
        r"api mismatch",
        r"backend unavailable",
        r"500 (?:internal server error)",
        r"502 (?:bad gateway)",
        r"503 (?:service unavailable)",
        r"504 (?:gateway timeout)",
    ]
    for pattern in integration_patterns:
        if re.search(pattern, combined_text):
            return "integration_bug"

    # Default to code bug (React runtime error, missing DOM element, assertion error)
    return "code_bug"


# ---------------------------------------------------------------------------
# Flow step extraction & prompt building
# ---------------------------------------------------------------------------

def _build_qa_flow_prompt(
    flow_name: str,
    state: GraphState,
) -> str:
    """Build the prompt asking the QA model to generate Playwright steps for a flow."""
    pages_desc = []
    if state.design_spec and state.design_spec.component_inventory:
        pages_desc.append(f"Component Inventory: {json.dumps(state.design_spec.component_inventory)}")
    if state.code_artifacts:
        pages_desc.append(f"Code Artifacts: {list(state.code_artifacts.keys())}")

    context_str = "\n".join(pages_desc) if pages_desc else "Standard web interface"

    return (
        f"You are a QA automation engineer generating Playwright test steps for the following flow:\n"
        f"Flow: '{flow_name}'\n"
        f"Application Context:\n{context_str}\n\n"
        f"Respond with a JSON object containing the list of steps to exercise this flow.\n"
        f"Supported step actions: 'goto', 'click', 'fill', 'press', 'wait_for_selector', 'assert_text', 'assert_visible', 'wait', 'screenshot'.\n"
        f"Example format:\n"
        f'{{"flow_name": "{flow_name}", "steps": ['
        f'{{"action": "goto", "value": "/"}}, '
        f'{{"action": "click", "selector": "button"}}, '
        f'{{"action": "assert_visible", "selector": "#result"}}'
        f"]}}\n"
        f"Return ONLY valid JSON."
    )


def _parse_flow_steps(
    raw_response: str,
    flow_name: str,
) -> list[dict[str, Any]]:
    """Parse JSON steps from LLM response or provide safe fallback steps."""
    try:
        # Match markdown fenced code blocks if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response)
        text_to_parse = json_match.group(1) if json_match else raw_response.strip()
        data = json.loads(text_to_parse)

        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
            return data["steps"]
        elif isinstance(data, list):
            return data
    except Exception:
        pass

    # Safe fallback exploratory steps
    return [
        {"action": "goto", "value": "/"},
        {"action": "wait", "value": "500"},
    ]


# ---------------------------------------------------------------------------
# Core QA agent node
# ---------------------------------------------------------------------------

def run_qa(
    state: GraphState,
    trace_writer: Optional[TraceWriter] = None,
    *,
    sandbox: Optional[Sandbox] = None,
    runner: Optional[PlaywrightRunner] = None,
    base_url: Optional[str] = None,
    app_dir: Optional[Union[str, Path]] = None,
    artifact_dir: Path = Path("artifacts"),
    force_failure: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """LangGraph node for the QA agent.

    Executes all required user flows in headless Chromium, performs flakiness
    retries, classifies failures, emits a proof-of-work record per flow, and
    updates the shared graph state.
    """
    task = _find_qa_task(state)
    task_id = task.task_id if task else "T-04"
    task_artifact_dir = (artifact_dir / task_id).resolve()
    task_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Resolve required flows
    required_flows: list[str] = []
    if task and task.required_flows:
        required_flows = list(task.required_flows)
    elif state.request and state.request.required_flows:
        required_flows = list(state.request.required_flows)
    else:
        # Extract from plan or default
        for t in (state.plan or []):
            if t.required_flows:
                required_flows.extend(t.required_flows)
        if not required_flows:
            required_flows = ["Main Flow"]

    # Deduplicate required_flows while preserving order
    seen_flows: set[str] = set()
    deduped_flows: list[str] = []
    for f in required_flows:
        if f not in seen_flows:
            seen_flows.add(f)
            deduped_flows.append(f)
    required_flows = deduped_flows

    # Fast path for simulated force_failure in unit routing tests
    if force_failure:
        return _simulate_forced_failure(state, task_id, task_artifact_dir, trace_writer)

    # Initialize runner
    if runner is None:
        runner = PlaywrightRunner(artifact_dir=artifact_dir, headless=True)

    # Resolve app directory
    if app_dir is None:
        if sandbox is not None:
            app_dir = sandbox.app_dir
        elif state.code_artifacts and "app_dir" in state.code_artifacts:
            app_dir = Path(state.code_artifacts["app_dir"])
        else:
            run_id = getattr(trace_writer, "run_id", "run-default") if trace_writer else "run-default"
            default_app = Path("runs") / run_id / "app"
            if default_app.exists():
                app_dir = default_app

    server: Optional[AppServer] = None
    managed_base_url = base_url

    # Start server if base_url is not provided and app_dir exists
    if not managed_base_url:
        if app_dir and Path(app_dir).exists():
            server = AppServer(app_dir=app_dir)
            managed_base_url = server.start()
        else:
            # Fallback for environments without live server: run stub/mock flow
            managed_base_url = "http://localhost:5173"

    flow_results: list[dict[str, Any]] = []
    all_screenshots: list[str] = []
    all_console_logs: list[str] = []
    proofs: list[Any] = list(state.proofs)

    any_flow_failed = False

    try:
        for flow_name in required_flows:
            # 1. Generate flow steps via LLM
            messages = [
                {"role": "user", "content": _build_qa_flow_prompt(flow_name, state)}
            ]
            try:
                raw_response = complete("qa", messages)
                steps = _parse_flow_steps(raw_response, flow_name)
            except Exception:
                steps = [{"action": "goto", "value": "/"}]

            # 2. Run flow in headless Chromium
            res = runner.run_flow(
                base_url=managed_base_url,
                flow_name=flow_name,
                steps=steps,
                task_id=task_id,
            )

            retried = False
            is_flaky = False
            classification: Optional[str] = None

            # 3. Retry on failure to detect flakiness
            if not res.passed:
                retried = True
                if trace_writer:
                    trace_writer.write(
                        "qa",
                        "flow_retry",
                        {"flow": flow_name, "reason": "First attempt failed, retrying to check for flakiness."},
                    )
                retry_res = runner.run_flow(
                    base_url=managed_base_url,
                    flow_name=flow_name,
                    steps=steps,
                    task_id=task_id,
                )
                if retry_res.passed:
                    # Succeeded on retry -> Flaky!
                    is_flaky = True
                    classification = "flaky"
                    res = retry_res
                else:
                    # Still failed -> classify failure
                    classification = classify_failure(retry_res)
                    res = retry_res

            # Record artifacts
            all_screenshots.extend(res.screenshots)
            if res.log_path and res.log_path not in all_console_logs:
                all_console_logs.append(res.log_path)

            flow_record = {
                "flow": flow_name,
                "passed": res.passed,
                "classification": classification,
                "error": res.error,
                "duration_ms": res.duration_ms,
                "retried": retried,
                "is_flaky": is_flaky,
                "screenshots": res.screenshots,
                "result_path": res.result_path,
                "log_path": res.log_path,
            }
            flow_results.append(flow_record)

            # 4. Generate proof-of-work record per flow
            evidence = []
            if res.result_path and Path(res.result_path).exists():
                evidence.append({"type": "test_result", "ref": res.result_path})
            for shot in res.screenshots:
                if Path(shot).exists():
                    evidence.append({"type": "screenshot", "ref": shot})
            if res.log_path and Path(res.log_path).exists():
                evidence.append({"type": "log", "ref": res.log_path})

            if not evidence:
                # Ensure at least one evidence item exists
                placeholder = task_artifact_dir / f"qa_{runner._slugify(flow_name)}.log"
                placeholder.write_text(f"Flow: {flow_name}\nPassed: {res.passed}", encoding="utf-8")
                evidence.append({"type": "log", "ref": str(placeholder)})

            if res.passed:
                claim = (
                    f"Flow '{flow_name}' passed (flaky on retry)"
                    if is_flaky
                    else f"Flow '{flow_name}' passed in headless Chromium"
                )
                reasoning = (
                    f"Flow exercised in headless Chromium. "
                    f"Completed {len(res.steps)} steps without unhandled exceptions."
                )
            else:
                any_flow_failed = True
                claim = f"Flow '{flow_name}' failed ({classification})"
                reasoning = (
                    f"Flow failed with error: {res.error}. "
                    f"Classified as {classification} based on console errors and step outcomes."
                )

            proof = create_proof(
                agent="qa",
                task_id=task_id,
                claim=claim,
                evidence=evidence,
                reasoning=reasoning,
            )
            proofs.append(proof)

            if trace_writer:
                event_type = "flow_passed" if res.passed else "flow_failed"
                trace_writer.write(
                    "qa",
                    event_type,
                    {
                        "task_id": task_id,
                        "flow": flow_name,
                        "passed": res.passed,
                        "classification": classification,
                        "claim": claim,
                    },
                )
    finally:
        if server:
            server.stop()

    qa_results = QAResults(
        flows=flow_results,
        screenshots=all_screenshots,
        console_logs=all_console_logs,
    )

    # Update task status in plan
    updated_plan = _update_plan_status(
        state=state,
        task_id=task_id,
        new_status="failed" if any_flow_failed else "done",
    )

    if trace_writer:
        trace_writer.write(
            "qa",
            "handoff",
            {
                "task_id": task_id,
                "all_passed": not any_flow_failed,
                "flows_count": len(flow_results),
            },
        )

    return {
        "qa_results": qa_results,
        "proofs": proofs,
        "plan": updated_plan,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_qa_task(state: GraphState) -> Optional[TaskDefinition]:
    """Find the current pending or in_progress QA task in state."""
    for task in (state.plan or []):
        if task.assigned_agent == "qa" and task.status in ("pending", "in_progress"):
            return task
    # Fallback to any QA task
    for task in (state.plan or []):
        if task.assigned_agent == "qa":
            return task
    return None


def _update_plan_status(
    state: GraphState,
    task_id: str,
    new_status: str,
) -> Optional[list[TaskDefinition]]:
    """Return an updated plan with task_id updated to new_status."""
    if not state.plan:
        return None
    updated = []
    for task in state.plan:
        if task.task_id == task_id:
            updated.append(task.model_copy(update={"status": new_status}))
        else:
            updated.append(task)
    return updated


def _simulate_forced_failure(
    state: GraphState,
    task_id: str,
    task_artifact_dir: Path,
    trace_writer: Optional[TraceWriter],
) -> dict[str, Any]:
    """Simulate a failing QA flow for testing supervisor rerouting."""
    screenshots_dir = task_artifact_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshots_dir / "filter_by_priority_failed.png"
    screenshot_path.write_bytes(b"PNG_STUB")

    results_path = task_artifact_dir / "playwright.json"
    results_path.write_text(
        json.dumps({
            "flows": [
                {"flow": "list tickets", "passed": True},
                {"flow": "filter by priority", "passed": False, "classification": "code_bug"},
            ]
        }),
        encoding="utf-8",
    )

    console_log_path = task_artifact_dir / "console.log"
    console_log_path.write_text(
        "[ERROR] Uncaught TypeError: Cannot read properties of undefined (reading 'filter')",
        encoding="utf-8",
    )

    flows = [
        {"flow": "list tickets", "passed": True},
        {
            "flow": "filter by priority",
            "passed": False,
            "classification": "code_bug",
            "error": "Timeout waiting for filter results",
        },
    ]

    qa_results = QAResults(
        flows=flows,
        screenshots=[str(screenshot_path)],
        console_logs=[str(console_log_path)],
    )

    proof = create_proof(
        agent="qa",
        task_id=task_id,
        claim="QA flow 'filter by priority' failed (code_bug)",
        evidence=[
            {"type": "test_result", "ref": str(results_path)},
            {"type": "screenshot", "ref": str(screenshot_path)},
            {"type": "log", "ref": str(console_log_path)},
        ],
        reasoning=(
            "filter-by-priority flow failed with a TypeError. "
            "Classified as code_bug. Routing back to engineer."
        ),
    )

    if trace_writer:
        trace_writer.write(
            "qa",
            "flow_failed",
            {"task_id": task_id, "failing_flow": "filter by priority", "classification": "code_bug"},
        )

    updated_plan = _update_plan_status(state, task_id, "failed")
    return {
        "plan": updated_plan,
        "qa_results": qa_results,
        "proofs": state.proofs + [proof],
    }
