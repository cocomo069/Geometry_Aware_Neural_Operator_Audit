"""Download the *preprocessed* AirfRANS dataset into ``data/raw/``.

Safety first: before a single byte of payload is fetched we issue an HTTP HEAD
(falling back to a 1-byte ranged GET for servers that do not answer HEAD) to
learn ``Content-Length``, print it, and abort if it exceeds ``--max-gb``
(default 20 GB).  ``docs/PLAN.md`` risk register: "size-check via HEAD; if
>15 GB stop and reassess".

The URL is *introspected out of the installed ``airfrans`` package* rather than
hard-coded, so that a package upgrade that moves the dataset cannot silently
leave us pointing at a stale mirror.  A hard-coded fallback is used only if the
introspection fails (e.g. the package is not installed yet).

Usage
-----
    .venv/Scripts/python.exe scripts/download_data.py --check-only
    .venv/Scripts/python.exe scripts/download_data.py --skip-if-present

Notes
-----
* ``OpenFOAM=False`` -> the cropped/preprocessed ``.vtu``/``.vtp`` release, which
  is what CONTEXT.md section 2 mandates.
* The zip unpacks to ``<root>/Dataset/`` containing ``manifest.json`` and one
  directory per simulation.
* Licence: ODbL-1.0.  We redistribute split manifests and code, never the data.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

# Used only when the installed package cannot be introspected.
FALLBACK_URLS = {
    False: "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
    True: "https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip",
}

GB = 1024 ** 3
DEFAULT_MAX_GB = 20.0
_USER_AGENT = "geom-aware-neural-operator/0.1 (+size-check)"


class SizeCheckError(RuntimeError):
    """Raised when the remote payload is larger than the configured ceiling."""


def discover_download_url(openfoam: bool = False) -> tuple[str, str]:
    """Introspect ``airfrans.dataset.download`` for the dataset URL.

    Returns
    -------
    (url, source) : tuple[str, str]
        ``source`` is ``"airfrans.dataset"`` when the URL was recovered from
        the installed package source and ``"fallback"`` otherwise.
    """
    try:
        import airfrans.dataset as af_dataset  # noqa: PLC0415
    except Exception:
        return FALLBACK_URLS[openfoam], "fallback"

    try:
        src = inspect.getsource(af_dataset.download)
    except (OSError, TypeError):
        return FALLBACK_URLS[openfoam], "fallback"

    urls = re.findall(r"['\"](https?://[^'\"]+\.zip)['\"]", src)
    if not urls:
        return FALLBACK_URLS[openfoam], "fallback"

    # airfrans.dataset.download branches `if OpenFOAM: <OF url> else: <url>`,
    # so the OpenFOAM URL is the first literal and the preprocessed one second.
    of_urls = [u for u in urls if "OF_" in u or "OpenFOAM" in u]
    plain = [u for u in urls if u not in of_urls]
    if openfoam:
        chosen = of_urls[0] if of_urls else urls[0]
    else:
        chosen = plain[0] if plain else urls[-1]
    return chosen, "airfrans.dataset"


def remote_size_bytes(url: str, timeout: float = 30.0) -> int | None:
    """Return ``Content-Length`` for ``url`` without downloading the body.

    Tries HEAD first; if the server rejects HEAD or omits ``Content-Length``,
    retries with ``Range: bytes=0-0`` and parses ``Content-Range``.

    Returns
    -------
    int or None
        Size in bytes, or ``None`` if the server would not tell us.
    """
    headers = {"User-Agent": _USER_AGENT}

    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None:
                return int(length)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        pass

    req = urllib.request.Request(
        url, method="GET", headers={**headers, "Range": "bytes=0-0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_range = resp.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[-1].strip()
                if total.isdigit():
                    return int(total)
            length = resp.headers.get("Content-Length")
            # A non-ranged 200 response means the whole body is coming; its
            # Content-Length is then the true size.
            if resp.status == 200 and length is not None:
                return int(length)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        pass

    return None


def check_size(
    url: str,
    max_gb: float = DEFAULT_MAX_GB,
    allow_unknown: bool = False,
    timeout: float = 30.0,
) -> int | None:
    """Print the remote size and raise :class:`SizeCheckError` if too large.

    Parameters
    ----------
    url : str
        Dataset zip URL.
    max_gb : float
        Ceiling in gibibytes.
    allow_unknown : bool
        If the server refuses to report a size, proceed anyway when ``True``,
        otherwise raise.
    """
    size = remote_size_bytes(url, timeout=timeout)
    if size is None:
        msg = (
            f"Server did not report a Content-Length for {url}. Cannot verify "
            f"the payload is under {max_gb:g} GB."
        )
        if not allow_unknown:
            raise SizeCheckError(msg + " Re-run with --allow-unknown-size to "
                                       "proceed anyway.")
        print(f"WARNING: {msg} Proceeding (--allow-unknown-size).")
        return None

    print(f"Remote size: {size} bytes = {size / GB:.2f} GiB  ({url})")
    if size > max_gb * GB:
        raise SizeCheckError(
            f"ABORT: remote payload is {size / GB:.2f} GiB, which exceeds the "
            f"{max_gb:g} GiB ceiling. Free up disk / raise --max-gb "
            f"deliberately before retrying."
        )
    return size


def dataset_present(root: Path, file_name: str = "Dataset") -> bool:
    """True when an unzipped dataset with a manifest already exists."""
    return (Path(root) / file_name / "manifest.json").is_file()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="download/unzip destination (default: %(default)s)",
    )
    parser.add_argument(
        "--file-name",
        default="Dataset",
        help="zip basename passed to airfrans (default: %(default)s)",
    )
    parser.add_argument(
        "--openfoam",
        action="store_true",
        help="download the RAW OpenFOAM release instead of the preprocessed "
        "one. CONTEXT.md section 2 mandates the preprocessed release; only "
        "use this deliberately.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help="abort if the remote payload exceeds this many GiB "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--allow-unknown-size",
        action="store_true",
        help="proceed even if the server will not report Content-Length",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="resolve the URL and print the remote size, then exit without "
        "downloading anything",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="exit successfully if <root>/<file-name>/manifest.json already "
        "exists",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="download the zip but do not extract it",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if args.skip_if_present and dataset_present(root, args.file_name):
        print(
            f"Dataset already present at "
            f"{(root / args.file_name).resolve()} -- nothing to do."
        )
        return 0

    url, source = discover_download_url(openfoam=args.openfoam)
    print(f"Dataset URL ({source}): {url}")
    if source == "fallback":
        print(
            "NOTE: the airfrans package could not be introspected; using the "
            "hard-coded fallback URL."
        )

    try:
        check_size(
            url,
            max_gb=args.max_gb,
            allow_unknown=args.allow_unknown_size,
        )
    except SizeCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.check_only:
        print("--check-only: size check passed, not downloading.")
        return 0

    try:
        import airfrans as af  # noqa: PLC0415
    except ImportError:
        print(
            "airfrans is not installed; cannot download. "
            "Install it into .venv first.",
            file=sys.stderr,
        )
        return 3

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {root.resolve()} (this is a multi-GB transfer)...")
    af.dataset.download(
        root=str(root),
        file_name=args.file_name,
        unzip=not args.no_unzip,
        OpenFOAM=args.openfoam,
    )

    if not args.no_unzip:
        if dataset_present(root, args.file_name):
            print(
                f"OK: manifest found at "
                f"{(root / args.file_name / 'manifest.json').resolve()}"
            )
        else:
            print(
                "WARNING: download finished but no manifest.json was found "
                f"under {(root / args.file_name).resolve()}. Inspect the "
                "extracted tree before building the cache.",
                file=sys.stderr,
            )
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
