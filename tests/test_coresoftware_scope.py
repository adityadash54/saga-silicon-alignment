"""Coresoftware expert scope: commits traced back from MakeMilleFiles.cc."""

from __future__ import annotations

import os
import subprocess

import pytest

from stdiag import coresoftware_expert as ce

FILES = {
    "tier1": "offline/packages/TrackerMillepedeAlignment/MakeMilleFiles.cc",
    "tier2": "offline/packages/trackbase/ClusterErrorPara.cc",
    "both": "offline/packages/TrackerMillepedeAlignment/AlignmentDefs.h",
    "outside": "offline/packages/tpc/TpcClusterizer.cc",
}


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgSign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, message, *names, date="2026-10-01T12:00:00"):
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + message + "\n"
                        if path.exists() else message + "\n")
        git(repo, "add", name)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgSign=false", "commit", "-q", "-m",
                    message], cwd=repo, check=True,
                   env={**os.environ, "GIT_AUTHOR_DATE": date,
                        "GIT_COMMITTER_DATE": date})
    return git(repo, "rev-parse", "--short", "HEAD")


@pytest.fixture
def scoped_repo(tmp_path):
    repo = tmp_path / "coresoftware"
    repo.mkdir()
    git(repo, "init", "-q")
    base = commit(repo, "base", *FILES.values(), date="2026-01-01T12:00:00")
    hashes = {
        "outside": commit(repo, "tpc change", FILES["outside"]),
        "tier2": commit(repo, "error change", FILES["tier2"]),
        "tier1": commit(repo, "ntp change", FILES["tier1"]),
        "both": commit(repo, "labels and errors", FILES["both"],
                       FILES["tier2"]),
    }
    return repo, base, hashes


@pytest.mark.parametrize("revision", ["range", "window"])
def test_commits_are_split_by_tier(scoped_repo, revision):
    repo, base, hashes = scoped_repo
    rev = ({"from": base, "to": "HEAD"} if revision == "range"
           else {"since": "2026-06-01", "until": "now"})
    found = ce.traced_commits(repo, rev)
    tier1 = [item["commit"] for item in found["tier 1"]]
    tier2 = [item["commit"] for item in found["tier 2"]]
    assert tier1 == [hashes["both"], hashes["tier1"]]       # newest first
    assert tier2 == [hashes["tier2"]]                       # no duplicate
    assert hashes["outside"] not in tier1 + tier2
    both = found["tier 1"][0]
    assert sorted(both["files"]) == [FILES["both"]]         # tier-1 files only
    assert both["subject"] == "labels and errors"


def test_window_end_excludes_later_commits(scoped_repo):
    repo, _, _ = scoped_repo
    found = ce.traced_commits(repo, {"since": "2025-01-01",
                                     "until": "2026-06-01"})
    assert [item["subject"] for item in found["tier 1"]] == ["base"]
    assert found["tier 2"] == []


def test_request_lists_scope_commits_and_not_found_rule(scoped_repo):
    repo, base, hashes = scoped_repo
    text = ce.build_request({"id": "F1"}, {"from": base, "to": "HEAD"}, {},
                            repo)
    assert "## Scope and commits along the MakeMilleFiles chain" in text
    assert f"- {hashes['tier1']} " in text and f"- {hashes['tier2']} " in text
    assert hashes["outside"] not in text
    assert "offline/packages/TrackerMillepedeAlignment" in text
    assert ce.NOT_FOUND in ce.EXPERT_PROMPT
    assert "{not_found}" not in ce.EXPERT_PROMPT


def test_bad_revision_gives_a_note_not_a_crash(scoped_repo):
    repo, _, _ = scoped_repo
    text = ce.traced_commits_section(repo, {"from": "nope", "to": "HEAD"})
    assert "could not be made" in text


def test_scope_paths_exist_in_the_real_checkout():
    if not (ce.DEFAULT_REPO / ".git").exists():
        pytest.skip("no coresoftware checkout")
    missing = [path for entries in ce.TRACE_SCOPE.values()
               for path, _ in entries if not (ce.DEFAULT_REPO / path).exists()]
    assert missing == []
