"""Stage original Rust NedoTokenizer and notices in a platform wheel."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", type=Path,
                        default=Path(os.environ.get("MERCAN_SRC", ROOT / ".native-work/MercanRuntime")))
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    runtime = args.runtime_source.expanduser().resolve()
    filename = ("nedo004_ffi.dll" if sys.platform == "win32" else
                "libnedo004_ffi.dylib" if sys.platform == "darwin" else "libnedo004_ffi.so")
    source = args.library or runtime / "runtime/nedo004-ffi/target/release" / filename
    source = source.expanduser().resolve()
    notices = {
        "NedoTokenizer-LICENSE.txt": runtime / "runtime/NedoTokenizer/LICENSE",
        "ZEMBEREK-NOTICE.txt": runtime / "runtime/NedoTokenizer/THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md",
        "APACHE-2.0.txt": runtime / "runtime/NedoTokenizer/APACHE-2.0.txt",
    }
    for path in (source, *notices.values()):
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing NedoTokenizer build payload: {path}")
    destination = ROOT / "src/ethosoftlib/nedo/lib"
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in {filename: source, **notices}.items():
        fd, temporary = tempfile.mkstemp(prefix=".nedo-", dir=destination)
        os.close(fd)
        try:
            shutil.copy2(path, temporary)
            os.replace(temporary, destination / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print("Staged original NedoTokenizer:", destination / filename)


if __name__ == "__main__":
    main()
