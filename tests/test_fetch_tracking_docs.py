"""Local tracking docs snapshot (no network: ACTS download is faked)."""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import sys

import pytest

from stdiag import fetch_tracking_docs as fetch
from stdiag import coresoftware_expert, tracking_expert

PAGE = ("<html><head><script>alert('x')</script><style>p{}</style></head>"
        "<body><h2 id='ssec-meapar'>Measurements</h2><p>z = y - f(x)</p>"
        "<a name='ssec-lincon'></a><p>Constraints</p></body></html>")


@pytest.fixture
def saved_page(tmp_path):
    path = tmp_path / "saved" / "draftman_page.html"
    path.parent.mkdir()
    path.write_text(PAGE)
    return path


@pytest.fixture
def fake_acts(monkeypatch):
    calls = []

    def fake(ref, destination):
        calls.append(ref)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "version_number").write_text("45.0.0\n")
        for part in fetch.ACTS_PATHS:
            (destination / part).mkdir(parents=True, exist_ok=True)
            (destination / part / "File.hpp").write_text("// header\n")
        return {"ref": ref, "commit": "a" * 40, "version_number": "45.0.0"}

    monkeypatch.setattr(fetch, "fetch_acts", fake)
    return calls


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fetch_tracking_docs", *args])
    fetch.main()


def test_first_build_with_acts_and_local_page(tmp_path, saved_page,
                                              fake_acts, monkeypatch):
    out = tmp_path / "docs"
    run_main(monkeypatch, "--out", str(out), "--update-acts",
             "--millepede-html", str(saved_page))
    text = (out / "millepede" / "draftman_page.txt").read_text()
    assert "{#ssec-meapar}" in text and "{#ssec-lincon}" in text
    assert "alert" not in text and "p{}" not in text
    sources = json.loads((out / "SOURCES.json").read_text())
    assert sources["acts"]["version_number"] == "45.0.0"
    assert sources["millepede"]["local_file"] == "draftman_page.html"
    assert len(sources["millepede"]["sha256"]) == 64
    assert fetch.status(out) == []


def test_rebuild_keeps_acts_without_download(tmp_path, saved_page,
                                             fake_acts, monkeypatch):
    out = tmp_path / "docs"
    run_main(monkeypatch, "--out", str(out), "--update-acts",
             "--millepede-html", str(saved_page))
    saved_page.write_text(PAGE.replace("Constraints", "Constraints v2"))
    run_main(monkeypatch, "--out", str(out),
             "--millepede-html", str(saved_page))
    assert fake_acts == ["sPHENIX"]          # downloaded once only
    assert (out / "acts" / "version_number").exists()
    sources = json.loads((out / "SOURCES.json").read_text())
    assert sources["acts"]["commit"] == "a" * 40
    assert "Constraints v2" in (out / "millepede" /
                                "draftman_page.txt").read_text()


@pytest.mark.parametrize("content", [None, "<html>some other page</html>"])
def test_bad_page_leaves_snapshot_untouched(tmp_path, saved_page, fake_acts,
                                            monkeypatch, content):
    out = tmp_path / "docs"
    run_main(monkeypatch, "--out", str(out), "--update-acts",
             "--millepede-html", str(saved_page))
    before = (out / "SOURCES.json").read_text()
    bad = tmp_path / "bad.html"
    if content is not None:
        bad.write_text(content)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--out", str(out), "--update-acts",
                 "--millepede-html", str(bad))
    assert (out / "SOURCES.json").read_text() == before
    assert fake_acts == ["sPHENIX"]          # nothing downloaded on failure


def test_missing_acts_is_reported(tmp_path, saved_page, monkeypatch):
    out = tmp_path / "docs"
    with pytest.raises(SystemExit, match="--update-acts"):
        run_main(monkeypatch, "--out", str(out),
                 "--millepede-html", str(saved_page))
    with pytest.raises(SystemExit) as check:
        run_main(monkeypatch, "--out", str(out), "--check")
    assert check.value.code == 1


def test_tracking_expert_has_no_network_tools():
    for session in (None, "abc"):
        command = tracking_expert.claude_command("claude", "m", "high",
                                                 session_id=session)
        tools = command[command.index("--tools") + 1]
        assert "Web" not in tools
        allowed = command[command.index("--allowedTools") + 1:
                          command.index("--disallowedTools")]
        assert not any("Web" in rule for rule in allowed)
        disallowed = command[command.index("--disallowedTools") + 1:
                             command.index("--permission-mode")]
        assert {"WebFetch", "WebSearch"} <= set(disallowed)
        prompt = command[command.index("--append-system-prompt") + 1]
        assert "no network access" in prompt
        assert "www.desy.de" not in prompt


def test_require_inputs_needs_a_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking_expert, "DOCS_DIR", tmp_path / "empty")
    with pytest.raises(SystemExit, match="incomplete"):
        tracking_expert.require_inputs(tmp_path)


# --- ACTS download paths (git is faked) --------------------------------------

def fake_clone(parts):
    """A clone_acts stand-in that creates the given ACTS parts."""
    def clone(ref, clone_dir):
        clone_dir.mkdir(parents=True)
        (clone_dir / "version_number").write_text("45.0.0\n")
        for part in parts:
            (clone_dir / part / "include").mkdir(parents=True)
            (clone_dir / part / "include" / "A.hpp").write_text("//\n")
            (clone_dir / part / "notes.txt").write_text("not kept\n")
    return clone


@pytest.fixture
def old_acts(tmp_path):
    destination = tmp_path / "docs" / "acts"
    destination.mkdir(parents=True)
    (destination / "OLD").write_text("previous copy\n")
    return destination


def test_clone_commands_use_cone_mode(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(fetch, "run",
                        lambda command, cwd=None: commands.append(command) or "")
    fetch.clone_acts("sPHENIX", tmp_path / "a")
    assert commands[0][:2] == ["git", "clone"] and "--branch" in commands[0]
    assert commands[1][:4] == ["git", "sparse-checkout", "set", "--cone"]
    commands.clear()
    commit = "b" * 40
    fetch.clone_acts(commit, tmp_path / "b")
    assert ["git", "fetch", "-q", "--depth", "1", "origin", commit] in commands
    assert ["git", "sparse-checkout", "set", "--cone", *fetch.ACTS_PATHS] \
        in commands
    assert commands[-1] == ["git", "checkout", "-q", "FETCH_HEAD"]


def test_failed_download_keeps_old_copy(old_acts, monkeypatch):
    def fail(command, cwd=None):
        raise subprocess.CalledProcessError(128, command,
                                            stderr="fatal: unable to access")
    monkeypatch.setattr(fetch, "run", fail)
    with pytest.raises(SystemExit, match="unchanged"):
        fetch.fetch_acts("sPHENIX", old_acts)
    assert (old_acts / "OLD").exists()
    assert not old_acts.with_name("acts.new").exists()


def test_unexpected_layout_keeps_old_copy(old_acts, monkeypatch):
    monkeypatch.setattr(fetch, "clone_acts", fake_clone(["docs", "Core"]))
    monkeypatch.setattr(fetch, "run", lambda command, cwd=None: "c" * 40)
    with pytest.raises(SystemExit, match="no files in Plugins/Root"):
        fetch.fetch_acts("sPHENIX", old_acts)
    assert (old_acts / "OLD").exists()
    assert not old_acts.with_name("acts.new").exists()


def test_successful_download_replaces_and_counts(old_acts, monkeypatch):
    monkeypatch.setattr(fetch, "clone_acts", fake_clone(fetch.ACTS_PATHS))
    monkeypatch.setattr(fetch, "run", lambda command, cwd=None: "c" * 40)
    source = fetch.fetch_acts("sPHENIX", old_acts)
    assert not (old_acts / "OLD").exists()
    assert source["files"] == {part: 1 for part in fetch.ACTS_PATHS}
    assert (old_acts / "Core" / "include" / "A.hpp").exists()
    assert not (old_acts / "Core" / "notes.txt").exists()


def test_check_reports_an_empty_acts_part(tmp_path, saved_page, fake_acts,
                                          monkeypatch):
    out = tmp_path / "docs"
    run_main(monkeypatch, "--out", str(out), "--update-acts",
             "--millepede-html", str(saved_page))
    shutil.rmtree(out / "acts" / "Plugins" / "Json")
    assert fetch.status(out) == [
        "ACTS parts without files: Plugins/Json; run with --update-acts"]


# --- read-only git rules (approximate matcher; the real check is Claude
# Code's own matching, to be tested by the planned permissions eval) ------

def permitted(command: str) -> bool:
    def matches(rules):
        return any(fnmatch.fnmatchcase(f"Bash({command})", rule)
                   for rule in rules)
    return (matches(coresoftware_expert.ALLOWED_TOOLS)
            and not matches(coresoftware_expert.DISALLOWED_TOOLS))


@pytest.mark.parametrize("command", [
    "git log --oneline -5", "git show abc123:offline/x.cc",
    "git diff abc def -- offline", "git log -S localMeas --oneline",
    "git blame -L 10,20 x.cc", "git branch -a", "git branch -r",
    "git branch --contains abc123", "git status", "git ls-files offline",
])
def test_read_only_git_commands_are_permitted(command):
    assert permitted(command)


@pytest.mark.parametrize("command", [
    "git log --output=/tmp/x -1", "git log -1 --output /tmp/x",
    "git show --output=/tmp/x HEAD", "git diff --output=/tmp/x",
    "git diff HEAD --output=/tmp/x",
    "git grep pattern", "git grep -Osh pattern", "git grep -iOsh pattern",
    "git branch -D main", "git branch new-branch",
    "git branch --list -D main", "git branch --contains abc -m x y",
    "git fetch origin", "git pull", "git clone https://example.org/r.git",
    "git checkout main", "git reset --hard",
    "curl https://example.org",
])
def test_write_execute_and_network_commands_are_refused(command):
    assert not permitted(command)


@pytest.fixture
def local_acts_repo(tmp_path):
    """A small git repo with the ACTS folder layout, served as file://."""
    repo = tmp_path / "acts_origin"
    for part in [*fetch.ACTS_PATHS, "Fatras", "Examples/Python"]:
        (repo / part / "include").mkdir(parents=True)
        (repo / part / "include" / "A.hpp").write_text("//\n")
    (repo / "version_number").write_text("45.0.0\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "init.defaultBranch=sPHENIX"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "acts"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    return repo.as_uri(), commit


@pytest.mark.parametrize("by_hash", [False, True])
def test_real_git_sparse_checkout(tmp_path, local_acts_repo, monkeypatch,
                                  by_hash):
    url, commit = local_acts_repo
    monkeypatch.setattr(fetch, "ACTS_REPO", url)
    destination = tmp_path / "docs" / "acts"
    source = fetch.fetch_acts(commit if by_hash else "sPHENIX", destination)
    assert source["commit"] == commit
    assert source["version_number"] == "45.0.0"
    assert source["files"] == {part: 1 for part in fetch.ACTS_PATHS}
    assert (destination / "Examples" / "Framework" / "include" / "A.hpp").exists()
    assert not (destination / "Fatras").exists()            # not kept
    assert not (destination / "Examples" / "Python").exists()
