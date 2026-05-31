#!/usr/bin/env python3
r"""
High-Performance Computing (HPC) Pipeline Orchestrator and Provenance Tracker.

This module executes a sequential shell script of pipeline commands, 
captures the complete standard output and standard error streams for 
reproducibility (provenance logging), and automatically archives all 
diagnostic vector graphics (.pdf) generated during the execution window.

Stream Multiplexing Architecture:
    The engine reads the subprocess standard output at the byte level to 
    circumvent POSIX buffering artifacts. It utilizes a state machine to detect 
    the mathematical signature of iterative numerical solver updates. Upon detection, 
    it injects carriage returns (`\r`) and ANSI line-clearing codes (`\033[K`) 
    to forcefully overwrite the terminal display—simulating a native progress 
    bar—while simultaneously excluding these ephemeral transient states from 
    the persistent log to guarantee analytical readability.
"""

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the orchestrator.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments specifying the execution script 
        and the logical name of the experiment.
    """
    parser = argparse.ArgumentParser(
        description="Execute pipeline commands and archive provenance logs."
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the shell script (.sh) containing the sequential pipeline commands."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Logical identifier for the experiment."
    )
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to monitor for newly generated artifacts."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/reports"),
        help="Base directory where the provenance archives will be stored."
    )
    return parser.parse_args()


def extract_commands(script_path: Path) -> List[str]:
    """
    Extract valid execution commands from a shell script, ignoring comments 
    and resolving line continuations.

    Parameters
    ----------
    script_path : Path
        The file system path to the shell script.

    Returns
    -------
    List[str]
        A strictly ordered list of command strings.

    Raises
    ------
    FileNotFoundError
        If the specified script cannot be located.
    """
    if not script_path.is_file():
        raise FileNotFoundError(f"Execution script '{script_path}' cannot be located.")

    commands: List[str] = []
    with open(script_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
    
    consolidated: List[str] = []
    buffer_str = ""
    for cmd in commands:
        if cmd.endswith("\\"):
            buffer_str += cmd[:-1] + " "
        else:
            buffer_str += cmd
            consolidated.append(buffer_str)
            buffer_str = ""

    return consolidated


def execute_and_stream(command: str, log_file: Path) -> int:
    """
    Execute a shell command, multiplexing standard output to the terminal 
    while applying a state machine to construct a clean, iteration-free provenance log.

    Parameters
    ----------
    command : str
        The shell command string to be evaluated.
    log_file : Path
        The file system path to the transcript log.

    Returns
    -------
    int
        The termination return code of the executed process.
    """
    with open(log_file, 'a', encoding='utf-8') as f_header:
        f_header.write(f"\n{'='*80}\n")
        f_header.write(f"COMMAND : {command}\n")
        f_header.write(f"TIME    : {datetime.now().isoformat()}\n")
        f_header.write(f"{'='*80}\n\n")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    line_buffer = bytearray()
    last_was_progress = False

    def process_line(buffer: bytearray, f_log) -> None:
        nonlocal last_was_progress
        if not buffer:
            return

        line_str = buffer.decode('utf-8', errors='replace')
        
        # Robust heuristic signature for Nested Sampling iterations
        is_progress = "iter:" in line_str and ("eff(%):" in line_str or "bound:" in line_str)

        if is_progress:
            # Calculate hardware terminal width to prevent line-wrapping corruption
            term_width = shutil.get_terminal_size((80, 20)).columns
            display_str = line_str[:term_width - 1]
            
            # \r returns cursor to start of line, \033[K clears the entire row
            sys.stdout.write('\r\033[K' + display_str)
            sys.stdout.flush()
            last_was_progress = True
            # Explicitly exclude writing the progress state to the log file
        else:
            # Handle standard deterministic outputs (e.g., Evidence, Diagnostics)
            if last_was_progress:
                sys.stdout.write('\n')
            
            sys.stdout.write(line_str + '\n')
            sys.stdout.flush()
            f_log.write(buffer + b'\n')
            f_log.flush()
            last_was_progress = False

    with open(log_file, 'ab') as f_log:
        if process.stdout is not None:
            while True:
                char = process.stdout.read(1)
                if not char:
                    # Resolve remaining buffer sequence on EOF
                    process_line(line_buffer, f_log)
                    if last_was_progress:
                        sys.stdout.write('\n')
                    break

                # Multiplex byte sequence based on POSIX line termination standards
                if char in (b'\r', b'\n'):
                    process_line(line_buffer, f_log)
                    line_buffer.clear()
                else:
                    line_buffer.extend(char)

    process.wait()
    with open(log_file, 'a', encoding='utf-8') as f_footer:
        f_footer.write(f"\n[PROCESS TERMINATED WITH EXIT CODE: {process.returncode}]\n")

    return process.returncode


def harvest_artifacts(
    source_dir: Path, 
    target_dir: Path, 
    start_timestamp: float, 
    extension: str = ".pdf"
) -> None:
    """
    Scan the monitored directory for artifacts generated or modified 
    during the execution window and archive them.
    """
    if not source_dir.is_dir():
        sys.stdout.write(f"Warning: Artifact source directory '{source_dir}' does not exist.\n")
        return

    artifacts_collected = 0
    for file_path in source_dir.rglob(f"*{extension}"):
        try:
            modification_time = file_path.stat().st_mtime
            if modification_time >= start_timestamp:
                destination = target_dir / file_path.name
                shutil.copy2(file_path, destination)
                artifacts_collected += 1
        except Exception as e:
            sys.stderr.write(f"Failed to access artifact {file_path}: {e}\n")

    sys.stdout.write(f"Archived {artifacts_collected} artifact(s) of type '{extension}'.\n")


def main() -> None:
    """
    Primary execution sequence for orchestration and provenance logging.
    """
    args = parse_arguments()

    try:
        commands = extract_commands(args.script)
        if not commands:
            raise ValueError("No valid execution commands found in the specified script.")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = args.output_dir / f"{timestamp_str}_{args.name}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        log_file = archive_dir / "execution_transcript.log"
        start_time = time.time()

        sys.stdout.write(f"--- HPC Orchestrator Initialized ---\n")
        sys.stdout.write(f"Experiment Name   : {args.name}\n")
        sys.stdout.write(f"Provenance Archive: {archive_dir}\n")
        sys.stdout.write(f"Total Commands    : {len(commands)}\n\n")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT IDENTIFIER : {args.name}\n")
            f.write(f"INITIALIZATION TIME   : {datetime.now().isoformat()}\n")
            f.write(f"PIPELINE SCRIPT       : {args.script.resolve()}\n")

        for idx, cmd in enumerate(commands, 1):
            sys.stdout.write(f"\nExecuting step [{idx}/{len(commands)}]...\n")
            return_code = execute_and_stream(cmd, log_file)
            
            if return_code != 0:
                sys.stderr.write(f"\n[!] Fatal Error: Command failed with exit code {return_code}.\n")
                sys.stderr.write("Aborting subsequent pipeline operations to prevent cascading failures.\n")
                break

        sys.stdout.write("\n--- Execution Sequence Terminated ---\n")
        sys.stdout.write("Harvesting generated diagnostic artifacts...\n")
        
        harvest_artifacts(args.artifact_dir, archive_dir, start_time, extension=".pdf")

        sys.stdout.write(f"\nProvenance tracking complete. All records archived at:\n{archive_dir.resolve()}\n")

    except Exception as e:
        sys.stderr.write(f"Orchestrator failed due to operational error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()