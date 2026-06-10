from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1"


def require_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing active-lock atomic contract: {needle!r}")


def main() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    require_contains(text, "[IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None")
    require_contains(text, "[IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None")
    require_contains(text, "Refusing to update active lock not owned by this launcher")
    require_contains(text, "Test-ActiveLockOwnedByThisLauncher")
    require_contains(text, "Not writing startup failure active lock for ${RunId}; active lock is absent or not owned by this launcher.")
    require_contains(text, "Not archiving active lock for ${RunId}; active lock is absent or not owned by this launcher.")

    forbidden = [
        r"Move-Item\s+-LiteralPath\s+\$Temp\s+-Destination\s+\$Full",
        r"\[IO\.File\]::Replace\(\$Temp,\s*\$Full",
    ]
    for pattern in forbidden:
        if re.search(pattern, text):
            raise SystemExit(f"active-lock final path must not use temp replacement pattern: {pattern}")

    catch_match = re.search(r"} catch \{(?P<body>.*?)\n}\n\n\[pscustomobject\]@{", text, re.S)
    if not catch_match:
        raise SystemExit("could not locate supervisor startup catch block")
    catch_body = catch_match.group("body")
    write_index = catch_body.find('Write-ActiveLock -Payload (New-ActiveLockPayload `')
    archive_index = catch_body.find('Move-ActiveLockArchive -ArchiveRunId $RunId -Reason "startup_failed_stopped"')
    if write_index < 0 or archive_index < 0:
        raise SystemExit("startup catch must still write/archive its own startup failure lock")
    first_owner_check = catch_body.find("if (Test-ActiveLockOwnedByThisLauncher)")
    second_owner_check = catch_body.find("if (Test-ActiveLockOwnedByThisLauncher)", first_owner_check + 1)
    if first_owner_check < 0 or first_owner_check > write_index:
        raise SystemExit("startup failure write must be guarded by active-lock ownership")
    if second_owner_check < 0 or second_owner_check > archive_index:
        raise SystemExit("startup failure archive must be guarded by active-lock ownership")

    print("dml_active_lock_atomic_contract_ok=true")


if __name__ == "__main__":
    main()
