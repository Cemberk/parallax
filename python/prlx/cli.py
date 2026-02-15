"""PRLX CLI — console entry point for the GPU differential debugger."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import _find_lib

try:
    from importlib.metadata import version as _meta_version
    PRLX_VERSION = _meta_version("prlx")
except Exception:
    PRLX_VERSION = "dev"

PRLX_BANNER = r"""
 ____  ____  _     __  __
|  _ \|  _ \| |    \ \/ /
| |_) | |_) | |     \  /
|  __/|  _ <| |___  /  \
|_|   |_| \_\_____|/_/\_\  v{version}

PRLX — GPU Differential Debugger
{gpu_line}
"""


def _detect_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().split("\n")[0]
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            sm = parts[1].replace(".", "")
            return f"Auto-detected: NVIDIA {name} (SM_{sm})"
    except FileNotFoundError:
        pass  # nvidia-smi not installed
    except subprocess.SubprocessError:
        pass  # nvidia-smi failed (no driver, no GPU)
    return "No GPU detected"


def print_banner():
    gpu_info = _detect_gpu()
    banner = PRLX_BANNER.format(
        version=PRLX_VERSION,
        gpu_line=f"(c) 2026 | {gpu_info}",
    )
    print(banner, file=sys.stderr)


def find_site_map(search_dir=None):
    if search_dir is None:
        search_dir = Path.cwd()
    else:
        search_dir = Path(search_dir)

    candidates = [
        search_dir / "prlx-sites.json",
        search_dir / "build" / "prlx-sites.json",
        search_dir / ".." / "build" / "prlx-sites.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    current = search_dir
    for _ in range(5):
        site_map = current / "prlx-sites.json"
        if site_map.exists():
            return site_map.resolve()
        current = current.parent

    return None


def detect_gpu_arch():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(".")
                if len(parts) == 2:
                    major, minor = int(parts[0]), int(parts[1])
                    return major * 10 + minor
    except FileNotFoundError:
        pass  # nvidia-smi not installed
    except subprocess.SubprocessError:
        pass  # nvidia-smi failed
    return None


def find_clang_for_pass(pass_lib):
    if pass_lib is None or not pass_lib.exists():
        return None, None

    try:
        result = subprocess.run(
            ["readelf", "-d", str(pass_lib)],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "libLLVM.so." in line:
                m = re.search(r"libLLVM\.so\.(\d+)", line)
                if m:
                    llvm_ver = int(m.group(1))
                    clang = f"clang++-{llvm_ver}"
                    clang_path = subprocess.run(
                        ["which", clang], capture_output=True, text=True,
                    )
                    if clang_path.returncode == 0:
                        return clang_path.stdout.strip(), llvm_ver
    except FileNotFoundError:
        pass  # readelf not installed
    except subprocess.SubprocessError:
        pass  # readelf failed

    # For bundled versioned pass (libPrlxPass.llvm20.so), extract from filename
    m = re.search(r"llvm(\d+)", pass_lib.name)
    if m:
        llvm_ver = int(m.group(1))
        clang = f"clang++-{llvm_ver}"
        import shutil
        if shutil.which(clang):
            return clang, llvm_ver

    return None, None


def find_best_ptxas_arch(gpu_cc):
    # Falls back to SM 90 if nvcc is unavailable
    try:
        result = subprocess.run(
            ["nvcc", "--list-gpu-arch"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            max_arch = 0
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("compute_"):
                    arch_str = line.replace("compute_", "")
                    if arch_str.isdigit():
                        arch = int(arch_str)
                        if arch <= gpu_cc and arch > max_arch:
                            max_arch = arch
            if max_arch > 0:
                return max_arch
    except FileNotFoundError:
        pass  # nvcc not installed
    except subprocess.SubprocessError:
        pass  # nvcc failed
    return 90


def cmd_diff(args):
    trace_a = Path(args.trace_a)
    trace_b = Path(args.trace_b)

    if not trace_a.exists():
        print(f"Error: Trace A not found: {trace_a}", file=sys.stderr)
        return 1
    if not trace_b.exists():
        print(f"Error: Trace B not found: {trace_b}", file=sys.stderr)
        return 1

    # Auto-detect session mode: if both paths are directories, infer --session
    session = getattr(args, "session", False)
    if trace_a.is_dir() and trace_b.is_dir():
        session = True

    differ = _find_lib.find_differ_binary()
    if not differ:
        print(
            "Error: prlx-diff binary not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    site_map = None
    if args.map:
        site_map = Path(args.map)
        if not site_map.exists():
            print(f"Warning: Site map not found: {site_map}", file=sys.stderr)
            site_map = None
    else:
        site_map = find_site_map(trace_a.parent)
        if site_map:
            print(f"Auto-detected site map: {site_map}")
            # Staleness check: warn if site map is older than both traces
            try:
                map_mtime = site_map.stat().st_mtime
                a_mtime = trace_a.stat().st_mtime
                b_mtime = trace_b.stat().st_mtime
                if map_mtime < min(a_mtime, b_mtime):
                    print(
                        "Warning: Site map is older than both traces — "
                        "it may be stale. Recompile or pass --map explicitly.",
                        file=sys.stderr,
                    )
            except OSError:
                pass
        else:
            print("Note: No site map found. Divergences will show site IDs only.")
            print("      To see source locations, provide --map prlx-sites.json")

    cmd = [str(differ), str(trace_a), str(trace_b)]

    if site_map and site_map.exists():
        cmd.extend(["--map", str(site_map)])
    if args.values:
        cmd.append("--values")
    if args.verbose:
        cmd.append("--verbose")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.lookahead:
        cmd.extend(["--lookahead", str(args.lookahead)])
    if args.max_shown:
        cmd.extend(["-n", str(args.max_shown)])
    if hasattr(args, "tui") and args.tui:
        cmd.append("--tui")
    if hasattr(args, "float") and getattr(args, "float", False):
        cmd.append("--float")
    if hasattr(args, "force") and args.force:
        cmd.append("--force")
    if session:
        cmd.append("--session")

    return subprocess.call(cmd)


def cmd_run(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    if args.output:
        trace_file = args.output
    else:
        trace_file = f"{binary.stem}.prlx"

    env = os.environ.copy()
    env["PRLX_TRACE"] = trace_file

    print(f"Running {binary} with tracing enabled...")
    print(f"Trace output: {trace_file}")

    cmd = [str(binary)] + args.binary_args
    result = subprocess.call(cmd, env=env)

    if result == 0 and Path(trace_file).exists():
        print(f"Trace generated: {trace_file}")

    return result


def cmd_check(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a = Path(tmpdir) / "trace_a.prlx"
        trace_b = Path(tmpdir) / "trace_b.prlx"

        print("Non-determinism check: running the same binary twice and diffing traces.")
        print("To compare different inputs, use: prlx run + prlx run + prlx diff\n")
        print("=== Run A ===")
        env = os.environ.copy()
        env["PRLX_TRACE"] = str(trace_a)
        cmd = [str(binary)] + args.binary_args
        result_a = subprocess.call(cmd, env=env)

        if result_a != 0:
            print(f"Error: Run A failed with exit code {result_a}",
                  file=sys.stderr)
            return result_a
        if not trace_a.exists():
            print("Error: Run A did not generate a trace file",
                  file=sys.stderr)
            return 1

        print("\n=== Run B ===")
        env["PRLX_TRACE"] = str(trace_b)
        result_b = subprocess.call(cmd, env=env)

        if result_b != 0:
            print(f"Error: Run B failed with exit code {result_b}",
                  file=sys.stderr)
            return result_b
        if not trace_b.exists():
            print("Error: Run B did not generate a trace file",
                  file=sys.stderr)
            return 1

        print("\n=== Differential Analysis ===")

        class DiffArgs:
            def __init__(self):
                self.trace_a = trace_a
                self.trace_b = trace_b
                self.map = args.map
                self.values = args.values
                self.verbose = args.verbose
                self.limit = args.limit
                self.lookahead = args.lookahead
                self.max_shown = args.max_shown
                self.tui = getattr(args, "tui", False)
                self.float = getattr(args, "float", False)
                self.force = getattr(args, "force", False)
                self.session = False

        return cmd_diff(DiffArgs())


def cmd_compile(args):
    source = Path(args.source)

    if not source.exists():
        print(f"Error: Source file not found: {source}", file=sys.stderr)
        return 1

    pass_lib = _find_lib.find_pass_plugin()
    if not pass_lib:
        print(
            "Error: LLVM pass not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    clang, llvm_ver = find_clang_for_pass(pass_lib)
    if not clang:
        print("Error: No matching clang found for LLVM pass.", file=sys.stderr)
        print("       Install clang matching your LLVM version "
              "(e.g., apt install clang-20).", file=sys.stderr)
        return 1

    gpu_cc = detect_gpu_arch()
    ptxas_arch = find_best_ptxas_arch(gpu_cc) if gpu_cc else 90

    if args.arch:
        ptxas_arch = args.arch

    sm_str = f"sm_{ptxas_arch}"

    if args.output:
        output = args.output
    else:
        output = source.stem

    include_dirs = _find_lib.find_include_dirs()
    if not include_dirs:
        print("Warning: Could not find prlx headers. "
              "Compilation may fail.", file=sys.stderr)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, prefix="prlx_unified_"
    ) as f:
        unified_path = f.name
        f.write("// Auto-generated unified compilation wrapper\n")
        f.write('#include "prlx_runtime.cu"\n')
        f.write('#include "prlx_host.cu"\n')
        f.write(f'#include "{source.resolve()}"\n')

    try:
        cmd = [
            clang,
            f"-fpass-plugin={pass_lib}",
            f"--cuda-gpu-arch={sm_str}",
            f"--cuda-include-ptx={sm_str}",
            "-g", "-O0",
        ]

        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])
        cmd.extend(["-I", str(source.resolve().parent)])

        for inc in (args.include or []):
            cmd.extend(["-I", inc])

        cmd.extend([unified_path, "-lcudart", "-lm", "-o", str(output)])

        if args.extra:
            cmd.extend(args.extra)

        if args.verbose:
            print(f"GPU: cc{gpu_cc or '?'}, compiling for "
                  f"{sm_str} + PTX (JIT forward-compat)")
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not args.verbose)

        if result.returncode != 0:
            if not args.verbose and result.stderr:
                sys.stderr.buffer.write(result.stderr)
            return result.returncode

        print(f"Compiled: {output} (instrumented, {sm_str}+PTX)")
        return 0

    finally:
        os.unlink(unified_path)


def cmd_session(args):
    subcmd = args.session_command
    if not subcmd:
        print("Usage: prlx session {capture,inspect,diff} ...", file=sys.stderr)
        return 1

    if subcmd == "capture":
        return cmd_session_capture(args)
    elif subcmd == "inspect":
        return cmd_session_inspect(args)
    elif subcmd == "diff":
        return cmd_session_diff(args)

    print(f"Unknown session subcommand: {subcmd}", file=sys.stderr)
    return 1


def cmd_session_capture(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else Path(f"{binary.stem}_session")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PRLX_SESSION"] = str(output_dir.resolve())

    cmd = [str(binary)] + (args.binary_args or [])
    print(f"Running {binary} with session capture to {output_dir}...")
    result = subprocess.call(cmd, env=env)

    manifest = output_dir / "session.json"
    if result == 0 and manifest.exists():
        print(f"Session captured: {output_dir}")
        import json
        with open(manifest) as f:
            launches = json.load(f)
        print(f"  {len(launches)} kernel launch(es) recorded")
    elif result == 0:
        print(f"Warning: Binary exited successfully but no session.json found in {output_dir}",
              file=sys.stderr)

    return result


def cmd_session_inspect(args):
    import json

    session_dir = Path(args.session_dir)
    manifest_path = session_dir / "session.json"

    if not manifest_path.exists():
        print(f"Error: No session.json in {session_dir}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        launches = json.load(f)

    print(f"Session: {session_dir}")
    print(f"Launches: {len(launches)}")
    print()

    # Table header
    print(f"{'#':<4} {'Kernel':<40} {'Grid':<16} {'Block':<16} {'File'}")
    print("-" * 100)

    for i, launch in enumerate(launches):
        grid = "x".join(str(g) for g in launch.get("grid", []))
        block = "x".join(str(b) for b in launch.get("block", []))
        kernel = launch.get("kernel", "?")
        fname = launch.get("file", "?")
        launch_idx = launch.get("launch", i)
        print(f"{launch_idx:<4} {kernel:<40} {grid:<16} {block:<16} {fname}")

    return 0


def cmd_session_diff(args):
    """Delegate to cmd_diff with session=True."""

    class DiffArgs:
        def __init__(self):
            self.trace_a = args.session_a
            self.trace_b = args.session_b
            self.map = getattr(args, "map", None)
            self.values = getattr(args, "values", False)
            self.verbose = getattr(args, "verbose", False)
            self.limit = getattr(args, "limit", None)
            self.lookahead = getattr(args, "lookahead", None)
            self.max_shown = getattr(args, "max_shown", 10)
            self.tui = getattr(args, "tui", False)
            self.float = getattr(args, "float", False)
            self.force = getattr(args, "force", False)
            self.session = True

    return cmd_diff(DiffArgs())


def cmd_triton(args):
    if args.info or args.check_env:
        print("=== PRLX Triton Integration ===")

        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            print(f"  LLVM Pass Plugin: {pass_plugin}")
        else:
            print("  LLVM Pass Plugin: NOT FOUND")

        runtime_lib = _find_lib.find_runtime_library()
        if runtime_lib:
            print(f"  Runtime Library:  {runtime_lib}")
        else:
            print("  Runtime Library:  NOT FOUND")

        bitcode = _find_lib.find_runtime_bitcode()
        if bitcode:
            print(f"  Runtime Bitcode:  {bitcode}")
        else:
            print("  Runtime Bitcode:  NOT FOUND")

        differ = _find_lib.find_differ_binary()
        if differ:
            print(f"  Differ Binary:    {differ}")
        else:
            print("  Differ Binary:    NOT FOUND")

        opt = _find_lib.find_opt_binary()
        if opt:
            print(f"  opt:              {opt}")
        else:
            print("  opt:              NOT FOUND (apt install llvm-20)")

        llvm_link = _find_lib.find_llvm_link_binary()
        if llvm_link:
            print(f"  llvm-link:        {llvm_link}")
        else:
            print("  llvm-link:        NOT FOUND (apt install llvm-20)")

        try:
            import triton
            print(f"  Triton:           {triton.__version__}")
        except ImportError:
            print("  Triton:           NOT INSTALLED (pip install triton)")

        try:
            import prlx
            print(f"  prlx package:     {prlx.__version__}")
        except ImportError:
            print("  prlx package:     NOT INSTALLED")

        print()
        print("Quick start:")
        print("  python -c 'import prlx; prlx.enable()'")
        return 0

    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}", file=sys.stderr)
            return 1

        env = os.environ.copy()

        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            existing = env.get("LLVM_PASS_PLUGINS", "")
            env["LLVM_PASS_PLUGINS"] = (
                f"{existing};{pass_plugin}" if existing else str(pass_plugin)
            )
        else:
            print("Warning: LLVM pass plugin not found", file=sys.stderr)

        cmd = [sys.executable, str(script_path)] + args.script_args
        return subprocess.call(cmd, env=env)

    print("Usage:")
    print("  prlx triton --info           Show integration status")
    print("  prlx triton script.py        Run script with instrumentation")
    return 0


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="PRLX - Differential debugger for CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two traces
  prlx diff trace_a.prlx trace_b.prlx

  # Compare with source location mapping
  prlx diff trace_a.prlx trace_b.prlx --map prlx-sites.json

  # Compare session directories (multi-launch traces)
  prlx diff session_a/ session_b/ --session

  # Run a binary with tracing
  prlx run ./my_kernel --output my_trace.prlx

  # Run twice and diff (detect non-determinism)
  prlx check ./my_kernel

  # Compile with automatic instrumentation
  prlx compile kernel.cu -o my_kernel

  # Triton integration info
  prlx triton --info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # diff
    p_diff = subparsers.add_parser("diff", help="Compare two trace files")
    p_diff.add_argument("trace_a", help="First trace file (baseline)")
    p_diff.add_argument("trace_b", help="Second trace file (compare)")
    p_diff.add_argument("--map", help="Site mapping file (prlx-sites.json)")
    p_diff.add_argument("-n", "--max-shown", type=int, default=10,
                        help="Max divergences to display")
    p_diff.add_argument("--values", action="store_true",
                        help="Compare operand values")
    p_diff.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    p_diff.add_argument("-l", "--limit", type=int,
                        help="Max divergences to collect")
    p_diff.add_argument("--lookahead", type=int,
                        help="Lookahead window size")
    p_diff.add_argument("--tui", action="store_true",
                        help="Launch interactive TUI viewer")
    p_diff.add_argument("--float", action="store_true",
                        help="Display snapshot operands as floats")
    p_diff.add_argument("--force", action="store_true",
                        help="Skip kernel name check (compare different variants)")
    p_diff.add_argument("--session", action="store_true",
                        help="Compare session directories (multi-launch traces)")

    # run
    p_run = subparsers.add_parser("run",
                                  help="Run a binary with tracing enabled")
    p_run.add_argument("binary", help="Binary to execute")
    p_run.add_argument("-o", "--output", help="Output trace file")
    p_run.add_argument("binary_args", nargs="*",
                       help="Arguments to pass to binary")

    # check
    p_check = subparsers.add_parser(
        "check",
        help="Run binary twice with identical inputs and diff (tests non-determinism)",
    )
    p_check.add_argument("binary", help="Binary to execute")
    p_check.add_argument("--map",
                         help="Site mapping file (prlx-sites.json)")
    p_check.add_argument("-n", "--max-shown", type=int, default=10,
                         help="Max divergences to display")
    p_check.add_argument("--values", action="store_true",
                         help="Compare operand values")
    p_check.add_argument("-v", "--verbose", action="store_true",
                         help="Verbose output")
    p_check.add_argument("-l", "--limit", type=int,
                         help="Max divergences to collect")
    p_check.add_argument("--lookahead", type=int,
                         help="Lookahead window size")
    p_check.add_argument("--tui", action="store_true",
                         help="Launch interactive TUI viewer")
    p_check.add_argument("--float", action="store_true",
                         help="Display snapshot operands as floats")
    p_check.add_argument("binary_args", nargs="*",
                         help="Arguments to pass to binary")

    # compile
    p_compile = subparsers.add_parser(
        "compile", help="Compile CUDA source with instrumentation")
    p_compile.add_argument("source", help="CUDA source file (.cu)")
    p_compile.add_argument("-o", "--output", help="Output binary name")
    p_compile.add_argument("-I", "--include", action="append",
                           help="Include directories")
    p_compile.add_argument("--arch", type=int,
                           help="CUDA SM architecture (e.g. 90)")
    p_compile.add_argument("-v", "--verbose", action="store_true",
                           help="Show compilation command")
    p_compile.add_argument("--extra", nargs="*",
                           help="Extra compiler flags")

    # session
    p_session = subparsers.add_parser(
        "session", help="Multi-kernel session operations")
    session_subs = p_session.add_subparsers(dest="session_command",
                                            help="Session subcommand")

    # session capture
    p_sess_capture = session_subs.add_parser(
        "capture", help="Capture a session (run binary with PRLX_SESSION)")
    p_sess_capture.add_argument("binary", help="Binary to execute")
    p_sess_capture.add_argument("-o", "--output",
                                help="Output session directory")
    p_sess_capture.add_argument("binary_args", nargs="*",
                                help="Arguments to pass to binary")

    # session inspect
    p_sess_inspect = session_subs.add_parser(
        "inspect", help="Inspect a session manifest")
    p_sess_inspect.add_argument("session_dir",
                                help="Session directory to inspect")

    # session diff
    p_sess_diff = session_subs.add_parser(
        "diff", help="Diff two session directories")
    p_sess_diff.add_argument("session_a", help="First session directory")
    p_sess_diff.add_argument("session_b", help="Second session directory")
    p_sess_diff.add_argument("--map", help="Site mapping file")
    p_sess_diff.add_argument("-n", "--max-shown", type=int, default=10,
                             help="Max divergences to display")
    p_sess_diff.add_argument("--values", action="store_true",
                             help="Compare operand values")
    p_sess_diff.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output")
    p_sess_diff.add_argument("-l", "--limit", type=int,
                             help="Max divergences to collect")
    p_sess_diff.add_argument("--lookahead", type=int,
                             help="Lookahead window size")
    p_sess_diff.add_argument("--float", action="store_true",
                             help="Display snapshot operands as floats")
    p_sess_diff.add_argument("--force", action="store_true",
                             help="Skip kernel name check")

    # triton
    p_triton = subparsers.add_parser("triton",
                                     help="Triton/Python integration")
    p_triton.add_argument("--info", action="store_true",
                          help="Show integration status")
    p_triton.add_argument("--check-env", action="store_true",
                          help="Verify environment setup")
    p_triton.add_argument("script", nargs="?",
                          help="Python script to run with Triton integration")
    p_triton.add_argument("script_args", nargs="*",
                          help="Arguments to pass to the script")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "diff": cmd_diff,
        "compile": cmd_compile,
        "run": cmd_run,
        "check": cmd_check,
        "session": cmd_session,
        "triton": cmd_triton,
    }
    handler = dispatch.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1
