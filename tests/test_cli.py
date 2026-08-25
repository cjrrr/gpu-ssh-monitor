import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_ssh_monitor.cli import (
    discover_hosts,
    group_hosts_by_endpoint,
    json_payload,
    parse_nvidia_smi,
    query_host,
    resolve_ssh_endpoint,
)


SAMPLE = "0, NVIDIA A100-SXM4-80GB, GPU-abc, 42, 87, 40960, 81920, 312.5, 400.0\n"


class ParserTests(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        gpu = parse_nvidia_smi("trainer", SAMPLE)[0]
        self.assertEqual(gpu.host, "trainer")
        self.assertEqual(gpu.index, 0)
        self.assertEqual(gpu.name, "NVIDIA A100-SXM4-80GB")
        self.assertEqual(gpu.utilization_pct, 87)
        self.assertEqual(gpu.memory_total_mib, 81920)

    def test_na_values(self):
        sample = "0, Tesla T4, GPU-x, 50, N/A, 1, 100, [N/A], 70\n"
        gpu = parse_nvidia_smi("host", sample)[0]
        self.assertIsNone(gpu.utilization_pct)
        self.assertIsNone(gpu.power_draw_w)

    def test_bad_field_count(self):
        with self.assertRaisesRegex(ValueError, "expected 9 fields"):
            parse_nvidia_smi("host", "0, GPU\n")


class ConfigTests(unittest.TestCase):
    def test_discovery_and_include(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "extra.conf").write_text("Host worker-2 *.wild\n  HostName worker-2.example.invalid\n")
            (root / "config").write_text(
                "Host worker-1 worker-1 !blocked\n"
                "  HostName worker-1.example.invalid\n"
                "Host *\n  User ubuntu\n"
                "Include extra.conf\n"
            )
            self.assertEqual(discover_hosts(root / "config"), ["worker-1", "worker-2"])

    @patch("gpu_ssh_monitor.cli.subprocess.run")
    def test_resolve_ssh_endpoint_ignores_user(self, run):
        run.return_value = subprocess.CompletedProcess(
            [], 0, "hostname worker.example.invalid\nport 2200\nuser alice\n", ""
        )
        self.assertEqual(resolve_ssh_endpoint("worker-alice", None), ("worker.example.invalid", "2200"))

    @patch("gpu_ssh_monitor.cli.resolve_ssh_endpoint")
    def test_group_same_host_and_port_once(self, resolve):
        endpoints = {
            "worker": ("worker.example.invalid", "22"),
            "worker_alice": ("worker.example.invalid", "22"),
            "other": ("other.example.invalid", "22"),
        }
        resolve.side_effect = lambda host, _config: endpoints[host]
        groups = group_hosts_by_endpoint(["worker", "worker_alice", "other"], None)
        self.assertEqual(groups, [("worker", ["worker", "worker_alice"]), ("other", ["other"])])


class QueryTests(unittest.TestCase):
    @patch("gpu_ssh_monitor.cli.subprocess.run")
    def test_query_success(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, SAMPLE, "")
        result = query_host("trainer", 3)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.gpus), 1)
        command = run.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-2], "trainer")

    @patch("gpu_ssh_monitor.cli.subprocess.run")
    def test_query_failure_is_result(self, run):
        run.return_value = subprocess.CompletedProcess([], 255, "", "ssh: host unreachable\n")
        result = query_host("offline", 1)
        self.assertEqual(result.error, "ssh: host unreachable")

    def test_json_shape(self):
        from datetime import datetime, timezone
        from gpu_ssh_monitor.cli import HostResult

        payload = json.loads(
            json_payload(
                [HostResult("h", parse_nvidia_smi("h", SAMPLE), 12, aliases=["h", "h_user"])],
                datetime.now(timezone.utc),
            )
        )
        self.assertEqual(payload["hosts"][0]["gpus"][0]["index"], 0)
        self.assertEqual(payload["hosts"][0]["aliases"], ["h", "h_user"])


if __name__ == "__main__":
    unittest.main()
