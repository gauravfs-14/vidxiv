"""One-line install commands for missing tools, chosen for the machine vidxiv runs on."""
from __future__ import annotations

import platform
import shutil

BREW_SETUP = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'

# tool -> package manager -> command. "*" is the fallback for that OS family.
COMMANDS = {
    "ffmpeg": {
        "brew": "brew install ffmpeg", "apt": "sudo apt update && sudo apt install -y ffmpeg", "dnf": "sudo dnf install -y ffmpeg-free",
        "pacman": "sudo pacman -S --noconfirm ffmpeg", "zypper": "sudo zypper install -y ffmpeg", "winget": "winget install Gyan.FFmpeg",
        "choco": "choco install -y ffmpeg", "scoop": "scoop install ffmpeg",
    },
    "node": {
        "brew": "brew install node", "apt": "sudo apt update && sudo apt install -y nodejs npm", "dnf": "sudo dnf install -y nodejs npm",
        "pacman": "sudo pacman -S --noconfirm nodejs npm", "zypper": "sudo zypper install -y nodejs npm", "winget": "winget install OpenJS.NodeJS.LTS",
        "choco": "choco install -y nodejs-lts", "scoop": "scoop install nodejs-lts",
    },
}
CLAUDE = {"Darwin": "curl -fsSL https://claude.ai/install.sh | bash", "Linux": "curl -fsSL https://claude.ai/install.sh | bash",
          "Windows": "irm https://claude.ai/install.ps1 | iex"}
NOTES = {
    "ffmpeg": "Needed for narration and for mastering the video's audio.",
    "node": "Needed for the video engine. Version 20 or newer.",
    "claude": "Then run claude once in a terminal to sign in.",
    "codex": "Then run codex once in a terminal to sign in.",
}


def system() -> str:
    return platform.system()  # Darwin, Linux, Windows


def manager() -> str | None:
    for name in {"Darwin": ("brew",), "Linux": ("apt", "dnf", "pacman", "zypper", "brew"), "Windows": ("winget", "scoop", "choco")}.get(system(), ()):
        if shutil.which({"apt": "apt-get"}.get(name, name)):
            return name
    return None


def command_for(tool: str) -> dict | None:
    """{command, note, prerequisite?} for a missing tool on this machine, or None when no one-liner applies."""
    sysname, pm = system(), manager()
    cmd, pre = None, None
    if tool in COMMANDS:
        cmd = COMMANDS[tool].get(pm or "")
        if cmd is None and sysname == "Darwin":
            cmd, pre = COMMANDS[tool]["brew"], BREW_SETUP  # Homebrew is the usual way on a Mac
    elif tool == "claude":
        cmd = CLAUDE.get(sysname)
    elif tool == "codex":
        cmd = "brew install codex" if pm == "brew" and sysname == "Darwin" else "npm install -g @openai/codex"
    if not cmd:
        return None
    out = {"command": cmd, "note": NOTES.get(tool, "")}
    if pre:
        out["prerequisite"] = {"command": pre, "note": "Homebrew is not installed. Run this first."}
    if tool == "codex" and cmd.startswith("npm") and not shutil.which("npm"):
        out["prerequisite"] = {"command": (command_for("node") or {}).get("command", ""), "note": "npm is not installed. Install Node.js first."}
    return out


def hints(missing: list[str]) -> dict:
    return {t: h for t in missing if (h := command_for(t))}
