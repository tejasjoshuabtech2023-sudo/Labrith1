"""
Isolated execution sandbox for generated code.

Each run gets its own directory (runs/<run_id>/app) scaffolded from a
Vite + React + TypeScript + Tailwind template.

Exposes:
- write_file(rel_path, content) -> Path
- read_file(rel_path) -> str
- run(cmd, timeout) -> CommandResult

Path escape is strictly prohibited and raises PathEscapeError (ValueError).
Command outputs are saved to artifacts/ so they can be cited as evidence.

Design Doc §7: "Sandboxed working directory per run"
Design Doc §13: "Sandboxed working directory per run; no network access for generated code beyond the demo API"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


class PathEscapeError(ValueError):
    """Raised when a path attempts to escape the sandbox directory."""


@dataclass
class CommandResult:
    """Result of running a command inside the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    log_path: Optional[str] = None


class Sandbox:
    """Isolated working directory and execution environment per run."""

    def __init__(
        self,
        run_id: str,
        base_dir: Union[str, Path] = Path("runs"),
        template_dir: Union[str, Path] = Path("templates/vite-react-ts-tailwind"),
        artifact_dir: Union[str, Path] = Path("artifacts"),
    ) -> None:
        self.run_id = str(run_id)
        self.base_dir = Path(base_dir).resolve()
        self.app_dir = (self.base_dir / self.run_id / "app").resolve()
        self.artifact_dir = (Path(artifact_dir).resolve() / self.run_id).resolve()
        self.template_dir = Path(template_dir).resolve()
        self._cmd_seq = 0

        self._scaffold()

    # -----------------------------------------------------------------------
    # Directory setup and scaffolding
    # -----------------------------------------------------------------------

    def _scaffold(self) -> None:
        """Create the app and artifact directories and copy template files."""
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Remove leftover generated components only on fresh scaffold
        if not (self.app_dir / "package.json").exists():
            components_dir = self.app_dir / "src" / "components"
            if components_dir.exists():
                shutil.rmtree(components_dir, ignore_errors=True)

        if not self.template_dir.exists():
            return

        # Copy template files to app_dir, linking node_modules for speed
        for item in self.template_dir.iterdir():
            dest = self.app_dir / item.name
            if item.name == "node_modules":
                if not dest.exists():
                    self._link_node_modules(item, dest)
            elif item.name in ("dist", ".git", ".vite", "__pycache__"):
                continue
            elif item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest)
            elif item.is_file():
                if not dest.exists():
                    shutil.copy2(item, dest)

    def _link_node_modules(self, src: Path, dest: Path) -> None:
        """Create a symlink or junction for node_modules to avoid massive file copying."""
        try:
            if sys.platform == "win32":
                try:
                    import _winapi
                    _winapi.CreateJunction(str(src.resolve()), str(dest.resolve()))
                except Exception:
                    os.symlink(src, dest, target_is_directory=True)
            else:
                os.symlink(src, dest, target_is_directory=True)
        except Exception:
            # If linking fails, fall back to copytree if necessary
            if not dest.exists():
                try:
                    shutil.copytree(src, dest)
                except Exception:
                    pass

    # -----------------------------------------------------------------------
    # Path isolation guards
    # -----------------------------------------------------------------------

    def _resolve_safe_path(self, rel_path: Union[str, Path]) -> Path:
        """Validate that *rel_path* resolves strictly inside the sandbox directory.

        Raises:
            PathEscapeError: If the resolved path escapes the sandbox directory.
        """
        if not rel_path:
            raise PathEscapeError("Path cannot be empty.")

        path_obj = Path(rel_path)
        app_dir_resolved = self.app_dir.resolve()

        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.app_dir / path_obj).resolve()

        try:
            resolved.relative_to(app_dir_resolved)
        except ValueError:
            raise PathEscapeError(
                f"Path '{rel_path}' escapes sandbox directory '{self.app_dir}'."
            ) from None

        return resolved

    # -----------------------------------------------------------------------
    # File I/O
    # -----------------------------------------------------------------------

    def write_file(self, rel_path: Union[str, Path], content: str) -> Path:
        """Write UTF-8 text content to a file inside the sandbox.

        Creates parent directories as needed. Refuses paths that escape.
        """
        dest = self._resolve_safe_path(rel_path)
        if dest == self.app_dir:
            raise PathEscapeError("Cannot write to sandbox root directory as a file.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return dest

    def read_file(self, rel_path: Union[str, Path]) -> str:
        """Read and return UTF-8 text content from a file inside the sandbox.

        Refuses paths that escape.
        """
        target = self._resolve_safe_path(rel_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {rel_path}")
        if target.is_dir():
            raise IsADirectoryError(f"Cannot read directory as file: {rel_path}")

        return target.read_text(encoding="utf-8")

    # -----------------------------------------------------------------------
    # Command execution
    # -----------------------------------------------------------------------

    def run(
        self,
        cmd: Union[str, list[str]],
        timeout: int = 60,
        log_name: Optional[str] = None,
    ) -> CommandResult:
        """Execute a command inside the sandbox app directory.

        Saves command output to artifacts/ so it can be cited as evidence.
        """
        self._cmd_seq += 1
        if log_name:
            safe_log_name = Path(log_name).name
        else:
            safe_log_name = f"command_{self._cmd_seq}.log"

        log_file = self.artifact_dir / safe_log_name
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # On Windows, normalize 'npm' to 'npm.cmd' if cmd is a string or list
        executable_cmd = cmd
        if sys.platform == "win32":
            if isinstance(cmd, str):
                trimmed = cmd.strip()
                if trimmed.startswith("npm ") or trimmed == "npm":
                    executable_cmd = "npm.cmd" + trimmed[3:]
                elif trimmed.startswith("npx ") or trimmed == "npx":
                    executable_cmd = "npx.cmd" + trimmed[3:]
            elif isinstance(cmd, list) and cmd:
                if cmd[0] == "npm":
                    executable_cmd = ["npm.cmd"] + cmd[1:]
                elif cmd[0] == "npx":
                    executable_cmd = ["npx.cmd"] + cmd[1:]

        use_shell = isinstance(executable_cmd, str)

        try:
            proc = subprocess.run(
                executable_cmd,
                cwd=str(self.app_dir),
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = (exc.stdout or "").decode("utf-8") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (
                f"Command timed out after {timeout} seconds.\n"
                + ((exc.stderr or "").decode("utf-8") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
            )
        except Exception as exc:
            exit_code = 1
            stdout = ""
            stderr = f"Execution failed: {exc}"

        # Write log output to artifacts
        combined_output = stdout
        if stderr:
            if combined_output:
                combined_output += "\n" + stderr
            else:
                combined_output = stderr

        log_file.write_text(combined_output, encoding="utf-8")

        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            log_path=str(log_file),
        )
