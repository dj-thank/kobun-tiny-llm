from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_run_completion", ROOT / "scripts" / "check_run_completion.py")
if spec is None or spec.loader is None:
    raise SystemExit("could not import check_run_completion.py")
crc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crc)


def main() -> None:
    train = crc.active_markers("training")
    supervision = crc.active_markers("supervision")
    for marker in (
        "kobun_llm.train",
        "train_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_cuda_colab_and_watch.py",
    ):
        if marker not in train:
            raise SystemExit(f"training_scope_missing={marker}")
    for marker in (
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
    ):
        if marker in train:
            raise SystemExit(f"training_scope_should_not_include_supervisor_marker={marker}")
        if marker not in supervision:
            raise SystemExit(f"supervision_scope_missing={marker}")
    locks = {path.as_posix() for path in crc.canonical_active_locks()}
    source = (ROOT / "scripts" / "check_run_completion.py").read_text(encoding="utf-8")
    for needle in (
        "startup_mutex_health",
        "active_old_japanese_0_1b_training.lock",
        "colab_active_old_japanese_0_1b_cuda",
        "startup mutex still exists during release completion gate",
    ):
        if needle not in source:
            raise SystemExit(f"run_completion_missing_startup_mutex_contract={needle}")
    if (ROOT / "logs" / "active_old_japanese_0_1b_training.lock").exists():
        expected = str(Path("logs") / "active_old_japanese_0_1b_training.lock").replace("\\", "/")
        if expected not in locks:
            raise SystemExit("run_completion_canonical_locks_omit_startup_mutex")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        logs = tmp_root / "logs"
        logs.mkdir()
        active = logs / "colab_active_old_japanese_0_1b_cuda.old_japanese_0_1b_cuda_20260513_000000.json"
        failed = logs / "colab_active_old_japanese_0_1b_cuda.old_japanese_0_1b_cuda_20260513_000001.failed_non_release.20260513_000002.json"
        finished = logs / "colab_active_old_japanese_0_1b_cuda.old_japanese_0_1b_cuda_20260513_000003.finished.20260513_000004.json"
        stale = logs / "colab_active_old_japanese_0_1b_cuda.old_japanese_0_1b_cuda_20260513_000005.stale.expired.20260513_000006.json"
        for path in (active, failed, finished, stale):
            path.write_text("{}", encoding="utf-8")
        previous = Path.cwd()
        try:
            import os

            os.chdir(tmp_root)
            scoped = {path.as_posix() for path in crc.canonical_active_locks()}
        finally:
            os.chdir(previous)
        if active.as_posix().removeprefix(tmp_root.as_posix() + "/") not in scoped:
            raise SystemExit("run_completion_canonical_locks_omit_active_colab_lease")
        for archive in (failed, finished, stale):
            rel = archive.as_posix().removeprefix(tmp_root.as_posix() + "/")
            if rel in scoped:
                raise SystemExit(f"run_completion_canonical_locks_treats_archive_as_active={rel}")
    print("run_completion_active_scope_ok=true")


if __name__ == "__main__":
    main()
