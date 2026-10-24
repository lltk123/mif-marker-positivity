"""Meaningful boundary checks for the publication guard; no biological data access."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "check_publication.py"
spec = importlib.util.spec_from_file_location("publication_guard", SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class PublicationBoundaryTests(unittest.TestCase):
    def test_forbidden_artifacts(self):
        paths = [
            "datasets/PhenoBIC-Data/source.tif",
            "analyses/run/weights/V1.pt",
            "analyses/run/predictions/all_cells.csv",
            ".ssh/id_ed25519",
            ".venv/lib/site.py",
            "analyses/Pan-M-Gold/README.md",
            "docs/database_password.txt",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertIsNotNone(guard.path_problem(path))

    def test_summary_and_sources_allowed(self):
        for path in [
            "README.md",
            "analyses/run/scripts/inference.py",
            "analyses/run/results/metrics.csv",
        ]:
            with self.subTest(path=path):
                self.assertIsNone(guard.path_problem(path))

    def test_secret_match_does_not_echo_value(self):
        secret = "ghp_" + "A" * 36
        findings = guard.secret_findings("first line\nvalue = " + secret)
        self.assertEqual(findings, [("GitHub token", 2)])
        self.assertNotIn(secret, str(findings))

    def test_plain_configuration_and_checksums_are_not_secrets(self):
        text = "weight_sha256 = '" + "f" * 64 + "'\npassword = read_password_file()"
        self.assertEqual(guard.secret_findings(text), [])

    def test_markdown_local_link_extraction(self):
        text = "[web](https://example.org) ![plot](figures/a.png) [section](#scope)"
        self.assertEqual(guard.local_links(text), ["figures/a.png"])


if __name__ == "__main__":
    unittest.main()
