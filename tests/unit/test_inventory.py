import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.inventory import InventoryProfile, build_inventory, is_generated_path, parse_cargo_workspace_members


class InventoryHelpersTest(unittest.TestCase):
    def test_generated_path_heuristics(self) -> None:
        self.assertTrue(is_generated_path("crates/proto/generated/types.pb.rs"))
        self.assertFalse(is_generated_path("crates/runtime/src/lib.rs"))

    def test_parse_cargo_workspace_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "Cargo.toml"
            manifest.write_text(
                '[workspace]\nmembers = ["crates/*", "examples/*"]\n',
                encoding="utf-8",
            )
            self.assertEqual(parse_cargo_workspace_members(manifest), ["crates/*", "examples/*"])

    def test_build_inventory_emits_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/*"]\n',
                encoding="utf-8",
            )
            (repo_root / "crates" / "demo" / "src").mkdir(parents=True)
            (repo_root / "crates" / "demo" / "Cargo.toml").write_text(
                '[package]\nname = "demo"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo_root / "crates" / "demo" / "src" / "lib.rs").write_text(
                "pub fn demo() {}\n",
                encoding="utf-8",
            )

            profile = InventoryProfile(
                repo="demo",
                expected_ref=None,
                analysis_surfaces=("crates",),
                build_commands=("cargo build --workspace",),
                test_commands=("cargo test --workspace",),
                notes=("demo repo",),
            )

            inventory = build_inventory(repo_root, profile)
            self.assertEqual(inventory["manifest"]["repo"], "demo")
            self.assertIn("language_mix", inventory["manifest"])
            self.assertIn("repo_map", inventory)
            crate_paths = [item["path"] for item in inventory["repo_map"]["crate_boundaries"]]
            self.assertIn("crates/demo", crate_paths)


if __name__ == "__main__":
    unittest.main()
