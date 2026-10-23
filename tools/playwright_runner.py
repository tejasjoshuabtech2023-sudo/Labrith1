"""
Playwright Runner Tool.

Starts the generated app (Vite preview, dev, or static server) from the sandbox,
executes user flows in headless Chromium using Playwright, captures screenshots,
console errors, and per-step results, and saves all evidence to artifacts/.

Design Doc §3: "Runs the app in a headless browser and exercises user flows. Tools: Playwright, screenshots, console log capture."
Design Doc §6: "Test gate: the generated app's own unit tests pass, and every required user flow passes in Playwright (QA evidence: results, screenshots)."
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FlowStepResult:
    """Result of an individual flow step."""

    step_num: int
    action: str
    target: Optional[str]
    status: str  # "passed" | "failed"
    duration_ms: float
    error: Optional[str] = None
    screenshot_path: Optional[str] = None


@dataclass
class PlaywrightFlowResult:
    """Result of running an entire user flow in Playwright."""

    flow_name: str
    passed: bool
    steps: list[FlowStepResult]
    duration_ms: float
    error: Optional[str] = None
    screenshots: list[str] = field(default_factory=list)
    console_logs: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    result_path: Optional[str] = None
    log_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_name": self.flow_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "screenshots": self.screenshots,
            "console_logs": self.console_logs,
            "page_errors": self.page_errors,
            "result_path": self.result_path,
            "log_path": self.log_path,
            "steps": [asdict(s) for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Server lifecycle management
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    """Find and return an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout: float = 20.0, interval: float = 0.3) -> bool:
    """Poll URL until it responds with HTTP 2xx/3xx/4xx or timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PlaywrightRunnerHealthCheck"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status < 500:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def stop_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and all its child processes cleanly across platforms."""
    if proc.poll() is not None:
        return

    pid = proc.pid
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except Exception:
            pass

    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass


class AppServer:
    """Manages the background server process for the sandboxed app."""

    def __init__(
        self,
        app_dir: Union[str, Path],
        port: Optional[int] = None,
        command: Optional[Union[str, list[str]]] = None,
        timeout: float = 25.0,
    ) -> None:
        self.app_dir = Path(app_dir).resolve()
        self.port = port or find_free_port()
        self.command = command
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{self.port}"

    def start(self) -> str:
        """Start the app server and wait for it to become ready."""
        if self.command:
            cmd = self.command
        else:
            # Auto-detect start command
            pkg_json = self.app_dir / "package.json"
            dist_dir = self.app_dir / "dist"
            if pkg_json.exists():
                if dist_dir.exists():
                    cmd = ["npm", "run", "preview", "--", "--port", str(self.port), "--host", "localhost"]
                else:
                    cmd = ["npm", "run", "dev", "--", "--port", str(self.port), "--host", "localhost"]
            else:
                cmd = [sys.executable, "-m", "http.server", str(self.port), "--bind", "localhost"]

        executable_cmd = cmd
        if sys.platform == "win32":
            if isinstance(cmd, list) and cmd:
                if cmd[0] == "npm":
                    executable_cmd = ["npm.cmd"] + cmd[1:]
                elif cmd[0] == "npx":
                    executable_cmd = ["npx.cmd"] + cmd[1:]
            elif isinstance(cmd, str):
                trimmed = cmd.strip()
                if trimmed.startswith("npm ") or trimmed == "npm":
                    executable_cmd = "npm.cmd" + trimmed[3:]

        use_shell = isinstance(executable_cmd, str)

        self.proc = subprocess.Popen(
            executable_cmd,
            cwd=str(self.app_dir),
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        ready = wait_for_server(self.base_url, timeout=self.timeout)
        if not ready:
            stdout_data = ""
            stderr_data = ""
            if self.proc:
                stop_process_tree(self.proc)
                try:
                    out, err = self.proc.communicate(timeout=1)
                    stdout_data = out or ""
                    stderr_data = err or ""
                except Exception:
                    pass
            raise RuntimeError(
                f"App server failed to respond at {self.base_url} within {self.timeout}s.\n"
                f"Stdout: {stdout_data}\nStderr: {stderr_data}"
            )

        return self.base_url

    def stop(self) -> None:
        """Stop the app server."""
        if self.proc:
            stop_process_tree(self.proc)
            self.proc = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Playwright Runner
# ---------------------------------------------------------------------------

class PlaywrightRunner:
    """Executes browser flows in headless Chromium, recording evidence."""

    def __init__(
        self,
        artifact_dir: Union[str, Path] = Path("artifacts"),
        headless: bool = True,
        slow_mo: float = 0,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.headless = headless
        self.slow_mo = slow_mo

    def _slugify(self, text: str) -> str:
        """Create a filesystem-safe slug from a string."""
        s = re.sub(r"[^\w\s-]", "", text).strip().lower()
        return re.sub(r"[-\s]+", "_", s) or "flow"

    def run_flow(
        self,
        base_url: str,
        flow_name: str,
        steps: list[dict[str, Any]],
        task_id: str = "T-QA",
        timeout_ms: int = 10000,
    ) -> PlaywrightFlowResult:
        """Run a sequence of flow steps in a new browser context.

        Args:
            base_url: Base URL where the app is being served.
            flow_name: Identifier/title of the user flow.
            steps: List of step dictionaries (action, selector, value, etc.).
            task_id: Task identifier used for organizing artifacts.
            timeout_ms: Timeout per step in milliseconds.

        Returns:
            PlaywrightFlowResult with step outcomes, screenshots, and logs.
        """
        flow_slug = self._slugify(flow_name)
        task_artifact_dir = self.artifact_dir / task_id
        screenshots_dir = task_artifact_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        console_logs: list[str] = []
        page_errors: list[str] = []
        step_results: list[FlowStepResult] = []
        screenshot_paths: list[str] = []

        overall_start = time.time()
        flow_passed = True
        flow_error: Optional[str] = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                base_url=base_url,
            )
            page = context.new_page()

            # Listen to console messages and page errors
            def on_console(msg: Any) -> None:
                console_logs.append(f"[{msg.type.upper()}] {msg.text}")

            def on_pageerror(err: Any) -> None:
                msg = str(err)
                page_errors.append(msg)
                console_logs.append(f"[PAGEERROR] {msg}")

            page.on("console", on_console)
            page.on("pageerror", on_pageerror)

            # Ensure starting on base URL if first step is not goto
            if not steps or steps[0].get("action") != "goto":
                try:
                    page.goto(base_url, timeout=timeout_ms, wait_until="domcontentloaded")
                except Exception as exc:
                    flow_passed = False
                    flow_error = f"Failed to load base URL {base_url}: {exc}"

            if flow_passed:
                for idx, step_dict in enumerate(steps, start=1):
                    action = step_dict.get("action", "")
                    selector = step_dict.get("selector")
                    value = step_dict.get("value")
                    expected_text = step_dict.get("text")
                    step_timeout = step_dict.get("timeout_ms", timeout_ms)

                    step_start = time.time()
                    step_passed = True
                    step_err: Optional[str] = None
                    step_screenshot: Optional[str] = None

                    try:
                        self._execute_step(
                            page=page,
                            base_url=base_url,
                            action=action,
                            selector=selector,
                            value=value,
                            expected_text=expected_text,
                            timeout_ms=step_timeout,
                        )
                    except Exception as exc:
                        step_passed = False
                        step_err = str(exc)
                        flow_passed = False
                        flow_error = f"Step {idx} ({action}) failed: {exc}"

                    step_dur = (time.time() - step_start) * 1000

                    # Capture screenshot if requested or on step failure
                    if not step_passed or step_dict.get("screenshot"):
                        shot_name = f"{flow_slug}_step{idx}_{'fail' if not step_passed else 'snap'}.png"
                        shot_path = screenshots_dir / shot_name
                        try:
                            page.screenshot(path=str(shot_path), full_page=True)
                            step_screenshot = str(shot_path)
                            screenshot_paths.append(step_screenshot)
                        except Exception:
                            pass

                    step_results.append(
                        FlowStepResult(
                            step_num=idx,
                            action=action,
                            target=selector or value or expected_text,
                            status="passed" if step_passed else "failed",
                            duration_ms=round(step_dur, 2),
                            error=step_err,
                            screenshot_path=step_screenshot,
                        )
                    )

                    if not step_passed:
                        break

            # Final flow screenshot (pass or fail)
            final_name = f"{flow_slug}_{'passed' if flow_passed else 'failed'}.png"
            final_path = screenshots_dir / final_name
            try:
                page.screenshot(path=str(final_path), full_page=True)
                if str(final_path) not in screenshot_paths:
                    screenshot_paths.append(str(final_path))
            except Exception:
                pass

            context.close()
            browser.close()

        total_dur = (time.time() - overall_start) * 1000

        # Save console logs to artifacts
        log_file = task_artifact_dir / f"console_{flow_slug}.log"
        log_content = "\n".join(console_logs) if console_logs else "(No console logs recorded)"
        log_file.write_text(log_content, encoding="utf-8")

        # Save test result JSON to artifacts
        result_file = task_artifact_dir / f"playwright_{flow_slug}.json"
        flow_result = PlaywrightFlowResult(
            flow_name=flow_name,
            passed=flow_passed,
            steps=step_results,
            duration_ms=round(total_dur, 2),
            error=flow_error,
            screenshots=screenshot_paths,
            console_logs=console_logs,
            page_errors=page_errors,
            result_path=str(result_file),
            log_path=str(log_file),
        )

        result_file.write_text(
            json.dumps(flow_result.to_dict(), indent=2),
            encoding="utf-8",
        )

        return flow_result

    def _execute_step(
        self,
        page: Page,
        base_url: str,
        action: str,
        selector: Optional[str] = None,
        value: Optional[str] = None,
        expected_text: Optional[str] = None,
        timeout_ms: int = 10000,
    ) -> None:
        """Dispatch a single step action on the Playwright page."""
        action = action.lower().strip()

        if action == "goto":
            url = value or selector or base_url
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

        elif action == "click":
            if not selector:
                raise ValueError("Step 'click' requires a 'selector'.")
            page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            page.click(selector, timeout=timeout_ms)

        elif action in ("fill", "type"):
            if not selector:
                raise ValueError(f"Step '{action}' requires a 'selector'.")
            val = str(value if value is not None else "")
            page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            page.fill(selector, val, timeout=timeout_ms)

        elif action == "press":
            if not selector:
                raise ValueError("Step 'press' requires a 'selector'.")
            key = str(value or "Enter")
            page.press(selector, key, timeout=timeout_ms)

        elif action in ("wait_for_selector", "assert_visible"):
            if not selector:
                raise ValueError(f"Step '{action}' requires a 'selector'.")
            page.wait_for_selector(selector, state="visible", timeout=timeout_ms)

        elif action == "assert_text":
            if not selector:
                raise ValueError("Step 'assert_text' requires a 'selector'.")
            expected = str(expected_text if expected_text is not None else (value or ""))
            page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            actual_text = page.text_content(selector, timeout=timeout_ms) or ""
            if expected not in actual_text:
                raise AssertionError(
                    f"Element '{selector}' text mismatch: expected substring '{expected}', got '{actual_text}'."
                )

        elif action == "wait":
            ms = int(value or 1000)
            page.wait_for_timeout(ms)

        elif action == "screenshot":
            # Handled by caller/step wrapper
            pass

        else:
            raise ValueError(f"Unsupported flow action: '{action}'.")
