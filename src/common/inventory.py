from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}

LANGUAGE_BY_EXTENSION = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".css": "CSS",
    ".go": "Go",
    ".h": "C/C++ Header",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".json": "JSON",
    ".jsx": "JSX",
    ".lock": "Lockfile",
    ".md": "Markdown",
    ".proto": "Protocol Buffers",
    ".py": "Python",
    ".rs": "Rust",
    ".sh": "Shell",
    ".sql": "SQL",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TSX",
    ".txt": "Text",
    ".yaml": "YAML",
    ".yml": "YAML",
}

ENTRYPOINT_FILENAMES = {
    "lib.rs",
    "main.rs",
    "mod.rs",
    "index.ts",
    "index.tsx",
    "main.ts",
    "main.tsx",
}


@dataclass(frozen=True)
class InventoryProfile:
    repo: str
    expected_ref: Optional[str]
    analysis_surfaces: Sequence[str]
    build_commands: Sequence[str]
    test_commands: Sequence[str]
    notes: Sequence[str]


def run_git(repo_root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root)] + list(args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def detect_language(path: Path) -> str:
    if path.name == "Dockerfile":
        return "Dockerfile"
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "Other")


def is_generated_path(relative_path: str) -> bool:
    generated_patterns = (
        "/generated/",
        "/gen/",
        "/dist/",
        "/target/",
        "/vendor/",
    )
    if any(pattern in f"/{relative_path}/" for pattern in generated_patterns):
        return True
    return relative_path.endswith((".pb.rs", "_pb2.py", "_grpc.py"))


def manifest_kind(path: str, payload: Optional[dict] = None, is_workspace: bool = False) -> str:
    name = Path(path).name
    if name == "Cargo.toml":
        return "cargo-workspace" if is_workspace else "cargo-package"
    if name == "package.json":
        workspaces = payload.get("workspaces") if isinstance(payload, dict) else None
        return "node-workspace" if workspaces else "node-package"
    if name == "pyproject.toml":
        return "python-project"
    return "manifest"


def parse_cargo_workspace_members(cargo_toml: Path) -> List[str]:
    text = cargo_toml.read_text(encoding="utf-8")
    workspace_match = re.search(r"\[workspace\](.*?)(?:\n\[|$)", text, re.S)
    if not workspace_match:
        return []
    block = workspace_match.group(1)
    members_match = re.search(r"members\s*=\s*\[(.*?)\]", block, re.S)
    if not members_match:
        return []
    return re.findall(r'"([^"]+)"', members_match.group(1))


def load_package_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_patterns(repo_root: Path, patterns: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(repo_root / pattern)))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                expanded.append(path.relative_to(repo_root).as_posix())
    return sorted(dict.fromkeys(expanded))


def collect_source_roots(repo_root: Path, package_roots: Iterable[str]) -> List[str]:
    roots: List[str] = []
    for package_root in package_roots:
        package_path = repo_root / package_root
        src_dir = package_path / "src"
        if src_dir.exists():
            roots.append(src_dir.relative_to(repo_root).as_posix())
        tests_dir = package_path / "tests"
        if tests_dir.exists():
            roots.append(tests_dir.relative_to(repo_root).as_posix())
        build_script = package_path / "build.rs"
        if build_script.exists():
            roots.append(build_script.relative_to(repo_root).as_posix())
    return sorted(dict.fromkeys(roots))


def collect_entrypoints(repo_root: Path, source_roots: Iterable[str]) -> List[str]:
    entrypoints: List[str] = []
    for source_root in source_roots:
        base = repo_root / source_root
        for filename in ENTRYPOINT_FILENAMES:
            candidate = base / filename
            if candidate.exists():
                entrypoints.append(candidate.relative_to(repo_root).as_posix())
    return sorted(dict.fromkeys(entrypoints))


def collect_analysis_surface_roots(repo_root: Path, analysis_surfaces: Sequence[str]) -> List[str]:
    roots: List[str] = []
    for surface in analysis_surfaces:
        surface_path = repo_root / surface
        if not surface_path.exists():
            continue
        if (surface_path / "src").exists():
            roots.append((surface_path / "src").relative_to(repo_root).as_posix())
        if (surface_path / "tests").exists():
            roots.append((surface_path / "tests").relative_to(repo_root).as_posix())
        if (surface_path / "build.rs").exists():
            roots.append((surface_path / "build.rs").relative_to(repo_root).as_posix())
        for child in sorted(surface_path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "src").exists():
                roots.append((child / "src").relative_to(repo_root).as_posix())
            if (child / "tests").exists():
                roots.append((child / "tests").relative_to(repo_root).as_posix())
            if (child / "build.rs").exists():
                roots.append((child / "build.rs").relative_to(repo_root).as_posix())
    return sorted(dict.fromkeys(roots))


def collect_files(repo_root: Path) -> Dict[str, object]:
    directories = set()
    files = []
    by_extension: Dict[str, Dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_language: Dict[str, Dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    generated_markers = []
    test_directories = set()

    for current_root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORE_DIRS)
        current_path = Path(current_root)
        relative_dir = current_path.relative_to(repo_root).as_posix() if current_path != repo_root else "."
        directories.add(relative_dir)

        parts = set(Path(relative_dir).parts)
        if "tests" in parts:
            test_directories.add(relative_dir)

        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink():
                continue

            relative_path = path.relative_to(repo_root).as_posix()
            extension = path.suffix.lower()
            size = path.stat().st_size
            language = detect_language(path)
            generated = is_generated_path(relative_path)

            file_record = {
                "path": relative_path,
                "size": size,
                "extension": extension,
                "language": language,
                "generated": generated,
                "content_hash": sha1_file(path),
            }
            files.append(file_record)

            if generated:
                generated_markers.append(
                    {
                        "path": relative_path,
                        "reason": "path_heuristic",
                    }
                )

            by_extension[extension or "<none>"]["files"] += 1
            by_extension[extension or "<none>"]["bytes"] += size
            by_language[language]["files"] += 1
            by_language[language]["bytes"] += size

    extension_rollup = [
        {
            "extension": extension,
            "files": counts["files"],
            "bytes": counts["bytes"],
        }
        for extension, counts in sorted(
            by_extension.items(),
            key=lambda item: (-item[1]["files"], item[0]),
        )
    ]

    language_mix = [
        {
            "language": language,
            "files": counts["files"],
            "bytes": counts["bytes"],
        }
        for language, counts in sorted(
            by_language.items(),
            key=lambda item: (-item[1]["files"], item[0]),
        )
    ]

    largest_files = sorted(files, key=lambda item: (-item["size"], item["path"]))[:20]

    return {
        "directories": sorted(path for path in directories if path != "."),
        "files": files,
        "language_mix": language_mix,
        "by_extension": extension_rollup,
        "largest_files": largest_files,
        "generated_markers": generated_markers,
        "test_directories": sorted(test_directories),
    }


def collect_manifests(repo_root: Path) -> Dict[str, object]:
    manifest_files = []
    package_roots = []
    crate_boundaries = []
    workspace_manifests = []

    for candidate in sorted(repo_root.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part in IGNORE_DIRS for part in candidate.parts):
            continue
        if candidate.name not in {"Cargo.toml", "package.json", "pyproject.toml"}:
            continue

        relative_path = candidate.relative_to(repo_root).as_posix()
        payload = None
        is_workspace = False

        if candidate.name == "Cargo.toml":
            members = parse_cargo_workspace_members(candidate)
            is_workspace = bool(members)
            if is_workspace:
                workspace_manifests.append(relative_path)
            crate_boundaries.append(
                {
                    "path": candidate.parent.relative_to(repo_root).as_posix() or ".",
                    "manifest": relative_path,
                }
            )
        elif candidate.name == "package.json":
            payload = load_package_json(candidate)

        kind = manifest_kind(relative_path, payload=payload, is_workspace=is_workspace)
        manifest_files.append(
            {
                "path": relative_path,
                "kind": kind,
            }
        )

        package_roots.append(
            {
                "path": candidate.parent.relative_to(repo_root).as_posix() or ".",
                "manifest": relative_path,
                "kind": kind,
            }
        )

    return {
        "dependency_manifests": manifest_files,
        "probable_package_roots": sorted(package_roots, key=lambda item: (item["path"], item["manifest"])),
        "crate_boundaries": sorted(crate_boundaries, key=lambda item: item["path"]),
        "workspace_manifests": sorted(workspace_manifests),
    }


def workspace_member_roots(repo_root: Path) -> Dict[str, List[str]]:
    cargo_members = []
    package_members = []

    cargo_manifest = repo_root / "Cargo.toml"
    if cargo_manifest.exists():
        cargo_members = expand_patterns(repo_root, parse_cargo_workspace_members(cargo_manifest))

    package_json = repo_root / "package.json"
    if package_json.exists():
        payload = load_package_json(package_json)
        workspaces = payload.get("workspaces", [])
        if isinstance(workspaces, list):
            package_members = expand_patterns(repo_root, workspaces)

    return {
        "cargo_members": cargo_members,
        "package_members": package_members,
    }


def git_metadata(repo_root: Path, expected_ref: Optional[str]) -> Dict[str, Optional[str]]:
    branch = run_git(repo_root, ["branch", "--show-current"])
    return {
        "path": str(repo_root),
        "git_ref": run_git(repo_root, ["rev-parse", "HEAD"]) or None,
        "git_branch": branch or "DETACHED",
        "git_remote": run_git(repo_root, ["remote", "get-url", "origin"]) or None,
        "git_dirty": "dirty" if run_git(repo_root, ["status", "--porcelain"]) else "clean",
        "expected_ref": expected_ref,
    }


def build_inventory(repo_root: Path, profile: InventoryProfile) -> Dict[str, object]:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    file_snapshot = collect_files(repo_root)
    manifest_snapshot = collect_manifests(repo_root)
    workspace_members = workspace_member_roots(repo_root)

    source_roots = collect_source_roots(
        repo_root,
        [
            item["path"]
            for item in manifest_snapshot["probable_package_roots"]
        ],
    )
    parser_roots = collect_analysis_surface_roots(repo_root, profile.analysis_surfaces)

    module_graph_seeds = {
        "analysis_surfaces": list(profile.analysis_surfaces),
        "workspace_manifests": manifest_snapshot["workspace_manifests"],
        "crate_roots": workspace_members["cargo_members"],
        "package_roots": workspace_members["package_members"],
        "source_roots": source_roots,
        "entrypoints": collect_entrypoints(repo_root, source_roots),
    }

    manifest = {
        "schema_version": "0.1.0",
        "repo": profile.repo,
        "generated_at": timestamp,
        "source": git_metadata(repo_root, profile.expected_ref),
        "language_mix": file_snapshot["language_mix"],
        "file_inventory": {
            "tracked_files": len(file_snapshot["files"]),
            "directories": len(file_snapshot["directories"]),
            "by_extension": file_snapshot["by_extension"],
            "largest_files": file_snapshot["largest_files"],
        },
        "module_graph_seeds": module_graph_seeds,
        "dependency_manifests": manifest_snapshot["dependency_manifests"],
        "test_commands": list(profile.test_commands),
        "build_commands": list(profile.build_commands),
        "parser_relevant_source_roots": sorted(dict.fromkeys(parser_roots + source_roots)),
        "notes": list(profile.notes),
    }

    repo_map = {
        "schema_version": "0.1.0",
        "repo": profile.repo,
        "generated_at": timestamp,
        "directories": [
            {
                "path": path,
                "depth": len(Path(path).parts),
            }
            for path in file_snapshot["directories"]
        ],
        "files": file_snapshot["files"],
        "probable_package_roots": manifest_snapshot["probable_package_roots"],
        "crate_boundaries": manifest_snapshot["crate_boundaries"],
        "generated_code_markers": file_snapshot["generated_markers"],
        "test_directories": file_snapshot["test_directories"],
    }

    return {
        "manifest": manifest,
        "repo_map": repo_map,
    }


def write_inventory(output_root: Path, repo: str, inventory: Dict[str, object]) -> None:
    repo_output = output_root / repo
    repo_output.mkdir(parents=True, exist_ok=True)

    for name, payload in inventory.items():
        target = repo_output / f"{name}.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
