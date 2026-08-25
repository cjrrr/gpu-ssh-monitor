from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


GPU_QUERY = (
    "index,name,uuid,temperature.gpu,utilization.gpu,"
    "memory.used,memory.total,power.draw,power.limit"
)
REMOTE_COMMAND = f"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits"
WILDCARDS = set("*?!")
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
CLEAR = "\033[H\033[2J"


@dataclass
class GPU:
    host: str
    index: int
    name: str
    uuid: str
    temperature_c: Optional[float]
    utilization_pct: Optional[float]
    memory_used_mib: Optional[float]
    memory_total_mib: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]


@dataclass
class HostResult:
    host: str
    gpus: List[GPU]
    latency_ms: int
    error: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


def _number(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_nvidia_smi(host: str, output: str) -> List[GPU]:
    gpus: List[GPU] = []
    for row_number, row in enumerate(csv.reader(output.splitlines()), start=1):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 9:
            raise ValueError(f"unexpected nvidia-smi row {row_number}: expected 9 fields, got {len(row)}")
        idx = _number(row[0])
        if idx is None:
            raise ValueError(f"invalid GPU index in row {row_number}")
        gpus.append(
            GPU(
                host=host,
                index=int(idx),
                name=row[1].strip(),
                uuid=row[2].strip(),
                temperature_c=_number(row[3]),
                utilization_pct=_number(row[4]),
                memory_used_mib=_number(row[5]),
                memory_total_mib=_number(row[6]),
                power_draw_w=_number(row[7]),
                power_limit_w=_number(row[8]),
            )
        )
    return gpus


def _strip_comment(line: str) -> str:
    # OpenSSH treats # as a comment outside quotes. This small parser is enough
    # for Host and Include directives while preserving quoted paths.
    quote: Optional[str] = None
    escaped = False
    result = []
    for char in line:
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\":
            result.append(char)
            escaped = True
        elif quote:
            result.append(char)
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            result.append(char)
            quote = char
        elif char == "#":
            break
        else:
            result.append(char)
    return "".join(result).strip()


def discover_hosts(config_path: Path, _visited: Optional[Set[Path]] = None) -> List[str]:
    visited = _visited if _visited is not None else set()
    path = config_path.expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    if path in visited or not path.is_file():
        return []
    visited.add(path)

    hosts: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []

    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].lower()
        if keyword == "host":
            for host in parts[1:]:
                if not any(char in host for char in WILDCARDS) and not host.startswith("!"):
                    hosts.append(host)
        elif keyword == "include":
            for pattern in parts[1:]:
                pattern = os.path.expanduser(pattern.strip("'\""))
                if not os.path.isabs(pattern):
                    pattern = str(path.parent / pattern)
                for included in sorted(glob.glob(pattern)):
                    hosts.extend(discover_hosts(Path(included), visited))
    return list(dict.fromkeys(hosts))


def resolve_ssh_endpoint(host: str, ssh_config: Optional[Path]) -> Tuple[str, str]:
    """Resolve an SSH alias to the hostname and port used for deduplication."""
    command = ["ssh", "-G"]
    if ssh_config is not None:
        command.extend(["-F", str(ssh_config.expanduser())])
    command.extend(["--", host])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("alias", host)
    if completed.returncode != 0:
        return ("alias", host)

    resolved = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator and key.lower() in {"hostname", "port"}:
            resolved[key.lower()] = value.strip()
    hostname = resolved.get("hostname")
    port = resolved.get("port")
    if not hostname or not port:
        return ("alias", host)
    return (hostname.casefold(), port)


def group_hosts_by_endpoint(hosts: Sequence[str], ssh_config: Optional[Path]) -> List[Tuple[str, List[str]]]:
    """Group aliases resolving to one endpoint while preserving config order."""
    groups = {}
    for host in hosts:
        endpoint = resolve_ssh_endpoint(host, ssh_config)
        groups.setdefault(endpoint, []).append(host)
    return [(aliases[0], aliases) for aliases in groups.values()]


def query_host(host: str, timeout: float, ssh_config: Optional[Path] = None) -> HostResult:
    connect_timeout = max(1, int(timeout))
    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={connect_timeout}",
        "-o", "ConnectionAttempts=1",
    ]
    if ssh_config is not None:
        command.extend(["-F", str(ssh_config.expanduser())])
    command.extend(["--", host, REMOTE_COMMAND])
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 1.0,
            check=False,
        )
        latency = round((time.monotonic() - started) * 1000)
        if completed.returncode != 0:
            message = completed.stderr.strip().splitlines()
            detail = message[-1] if message else f"ssh exited with code {completed.returncode}"
            return HostResult(host, [], latency, detail)
        try:
            gpus = parse_nvidia_smi(host, completed.stdout)
        except ValueError as exc:
            return HostResult(host, [], latency, str(exc))
        if not gpus:
            return HostResult(host, [], latency, "no NVIDIA GPUs reported")
        return HostResult(host, gpus, latency)
    except subprocess.TimeoutExpired:
        latency = round((time.monotonic() - started) * 1000)
        return HostResult(host, [], latency, f"timed out after {timeout:g}s")
    except OSError as exc:
        latency = round((time.monotonic() - started) * 1000)
        return HostResult(host, [], latency, str(exc))


def collect(hosts: Sequence[str], timeout: float, parallel: int, ssh_config: Optional[Path]) -> List[HostResult]:
    results = []
    with ThreadPoolExecutor(max_workers=min(parallel, len(hosts))) as pool:
        futures = {pool.submit(query_host, host, timeout, ssh_config): host for host in hosts}
        for future in as_completed(futures):
            results.append(future.result())
    order = {host: index for index, host in enumerate(hosts)}
    return sorted(results, key=lambda item: order[item.host])


def _fmt_number(value: Optional[float], suffix: str = "", decimals: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}{suffix}"


def _bar(percent: Optional[float], width: int = 10) -> str:
    if percent is None:
        return "-" * width
    bounded = max(0.0, min(100.0, percent))
    filled = round(bounded * width / 100)
    return "█" * filled + "░" * (width - filled)


def _color_for(value: Optional[float], yellow: float, red: float, enabled: bool) -> str:
    if not enabled or value is None:
        return ""
    if value >= red:
        return RED
    if value >= yellow:
        return YELLOW
    return GREEN


def render(results: Sequence[HostResult], color: bool, timestamp: datetime) -> str:
    ok = sum(result.error is None for result in results)
    gpu_count = sum(len(result.gpus) for result in results)
    alias_count = sum(len(result.aliases) if result.aliases else 1 for result in results)
    heading = (
        f"GPU SSH Monitor  {timestamp:%Y-%m-%d %H:%M:%S}  "
        f"{ok}/{len(results)} servers  {alias_count} SSH aliases  {gpu_count} GPUs"
    )
    if color:
        heading = BOLD + CYAN + heading + RESET
    lines = [heading]
    for result in results:
        if len(result.aliases) > 1:
            note = f"↳ {', '.join(result.aliases)} → queried once via {result.host}"
            lines.append((DIM + note + RESET) if color else note)
    lines.append("")
    header = f"{'HOST':<18} {'GPU':>3}  {'MODEL':<24} {'UTIL':>6} {'UTILIZATION':<10} {'MEMORY':>17} {'TEMP':>6} {'POWER':>13} {'RTT':>7}"
    lines.append((BOLD + header + RESET) if color else header)
    lines.append("─" * 115)
    for result in results:
        if result.error:
            prefix = RED if color else ""
            suffix = RESET if color else ""
            host_label = result.host + (f" [+{len(result.aliases) - 1}]" if len(result.aliases) > 1 else "")
            lines.append(f"{prefix}{host_label:<18}  !   {result.error} ({result.latency_ms}ms){suffix}")
            continue
        for position, gpu in enumerate(result.gpus):
            host = result.host + (f" [+{len(result.aliases) - 1}]" if len(result.aliases) > 1 else "")
            host = host[:18] if position == 0 else ""
            util_color = _color_for(gpu.utilization_pct, 70, 90, color)
            temp_color = _color_for(gpu.temperature_c, 70, 85, color)
            mem_pct = None
            if gpu.memory_used_mib is not None and gpu.memory_total_mib:
                mem_pct = 100 * gpu.memory_used_mib / gpu.memory_total_mib
            memory = f"{_fmt_number(gpu.memory_used_mib)}/{_fmt_number(gpu.memory_total_mib)} MiB"
            power = f"{_fmt_number(gpu.power_draw_w)}/{_fmt_number(gpu.power_limit_w)} W"
            util_text = _fmt_number(gpu.utilization_pct, "%")
            temp_text = _fmt_number(gpu.temperature_c, "°C")
            model = gpu.name[:24]
            line = (
                f"{host:<18} {gpu.index:>3}  {model:<24} "
                f"{util_color}{util_text:>6} {_bar(gpu.utilization_pct):<10}{RESET if util_color else ''} "
                f"{_color_for(mem_pct, 80, 95, color)}{memory:>17}{RESET if color and mem_pct is not None else ''} "
                f"{temp_color}{temp_text:>6}{RESET if temp_color else ''} "
                f"{power:>13} {result.latency_ms:>5}ms"
            )
            lines.append(line)
    return "\n".join(lines)


def json_payload(results: Sequence[HostResult], timestamp: datetime) -> str:
    payload = {
        "timestamp": timestamp.astimezone().isoformat(),
        "hosts": [
            {
                "host": result.host,
                "aliases": result.aliases or [result.host],
                "latency_ms": result.latency_ms,
                "error": result.error,
                "gpus": [asdict(gpu) for gpu in result.gpus],
            }
            for result in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsm",
        description="在一个终端中实时查看多台 SSH 服务器的 NVIDIA GPU 状态。",
    )
    parser.add_argument("hosts", nargs="*", help="SSH 主机名；省略时从 SSH 配置发现")
    parser.add_argument("-i", "--interval", type=float, default=2.0, help="刷新间隔秒数（默认：2）")
    parser.add_argument("-t", "--timeout", type=float, default=5.0, help="单台主机超时秒数（默认：5）")
    parser.add_argument("-p", "--parallel", type=int, default=16, help="最大并发查询数（默认：16）")
    parser.add_argument("-1", "--once", action="store_true", help="仅采集一次，不持续刷新")
    parser.add_argument("--json", action="store_true", help="输出一次 JSON（隐含 --once）")
    parser.add_argument("--match", metavar="REGEX", help="仅监控主机名匹配正则的主机")
    parser.add_argument("-F", "--ssh-config", type=Path, default=Path("~/.ssh/config"), help="SSH 配置路径")
    parser.add_argument("--no-config", action="store_true", help="连接时不向 ssh 传递 -F 配置路径")
    parser.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.1")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.interval <= 0:
        parser.error("--interval 必须大于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.parallel <= 0:
        parser.error("--parallel 必须大于 0")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    config_path = args.ssh_config.expanduser()
    hosts = list(dict.fromkeys(args.hosts or discover_hosts(config_path)))
    if args.match:
        try:
            matcher = re.compile(args.match)
        except re.error as exc:
            parser.error(f"--match 正则无效：{exc}")
        hosts = [host for host in hosts if matcher.search(host)]
    if not hosts:
        parser.error("没有可监控的主机；请传入主机名，或在 SSH 配置中添加明确的 Host 别名")

    once = args.once or args.json or not sys.stdout.isatty()
    color = not args.no_color and sys.stdout.isatty() and not args.json
    ssh_config = None if args.no_config else config_path
    host_groups = group_hosts_by_endpoint(hosts, ssh_config)
    query_hosts = [representative for representative, _aliases in host_groups]
    aliases_by_host = {representative: aliases for representative, aliases in host_groups}
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    first = True
    try:
        while not stop:
            results = collect(query_hosts, args.timeout, args.parallel, ssh_config)
            for result in results:
                result.aliases = aliases_by_host[result.host]
            now = datetime.now().astimezone()
            output = json_payload(results, now) if args.json else render(results, color, now)
            if not first and not once:
                sys.stdout.write(CLEAR)
            elif not once and sys.stdout.isatty():
                sys.stdout.write("\033[?25l" + CLEAR)
            sys.stdout.write(output + "\n")
            if not once:
                hint = "Ctrl-C 退出"
                sys.stdout.write((DIM + hint + RESET if color else hint) + "\n")
            sys.stdout.flush()
            first = False
            if once:
                return 1 if all(result.error for result in results) else 0
            deadline = time.monotonic() + args.interval
            while not stop and time.monotonic() < deadline:
                time.sleep(min(0.1, deadline - time.monotonic()))
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if not once and sys.stdout.isatty():
            sys.stdout.write("\033[?25h" + RESET + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
