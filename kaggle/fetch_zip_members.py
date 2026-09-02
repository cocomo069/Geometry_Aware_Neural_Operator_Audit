"""Extract ONLY specific members from a Kaggle kernel-output zip over HTTP Range.

PLAN_PHASE3 1.5 / 1.8. The Kaggle CLI's ``kernels_output`` does one un-chunked
``requests.get(url).content`` per file, buffering the whole thing in RAM with no
chunking, timeout, or resume -- a single 2 GB ``checkpoints.zip`` GET goes idle
and never completes on this connection. A zip's central directory lives at the
tail, so with HTTP Range requests we can read just the directory and then just
the bytes of the members we actually want (e.g. one run's ``best.pt``, ~16 MB out
of a 2 GB archive).

This is the recovery tool for the legacy fat ``checkpoints.zip`` (1.8) and the
fallback whenever ``pull_results.fetch_zip`` cannot get a member through the CLI.
It is best-effort: if Range is refused or the kaggle client cannot be reached, it
prints why and the caller falls back to the CLI (or the curl -C - loop in 1.8.3).

Usage:
  python kaggle/fetch_zip_members.py --kernel user/geo-op-session \
      --zip checkpoints.zip --pattern 'checkpoints/gnn_.*_s[1-4]/(best|last)\\.pt' \
      --dest .
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent


def kernel_output_file_url(kernel: str, filename: str) -> str:
    """Signed download URL for one file of a kernel's LATEST session output.

    Obtained exactly the way the CLI does, so it needs no extra auth beyond the
    usual kaggle.json. NOTE the CLI can only address the latest version's output
    (it never sends the version suffix), same limitation as ``kernels output``.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    owner, slug = kernel.split("/", 1)
    # Newer kaggle clients expose the typed session-output listing.
    try:
        from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore
            ApiListKernelSessionOutputRequest,
        )
        client = api.build_kaggle_client()
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
        for f in resp.files:
            if Path(f.url.split("?")[0]).name == filename or f.file_name == filename:
                return f.url
        raise SystemExit(f"{filename} not in {kernel} output listing")
    except ImportError:
        # Older client without the typed session-output request: the caller
        # falls back to the CLI (kaggle kernels output) or the curl -C - loop.
        raise SystemExit(
            "this kaggle client lacks ApiListKernelSessionOutputRequest; use the "
            "CLI (kaggle kernels output) or upgrade the kaggle package")


class HttpRangeFile(io.RawIOBase):
    """A minimal seekable file-like backed by HTTP Range requests.

    GCS-signed URLs (what Kaggle hands out) advertise ``Accept-Ranges: bytes``;
    the CLI's own ``download_file`` relies on that. ``zipfile.ZipFile`` reads the
    end-of-central-directory from the tail via ``seek``/``read``, so only the
    directory plus the requested members are ever transferred.
    """

    def __init__(self, url: str, refresh=None, timeout=60):
        import requests  # local import so the module imports without requests
        self._requests = requests
        self._url = url
        self._refresh = refresh          # callable -> fresh url on 403
        self._timeout = timeout
        self._pos = 0
        self._size = self._head_size()

    def _head_size(self) -> int:
        r = self._requests.get(self._url, headers={"Range": "bytes=0-0"},
                               timeout=self._timeout, stream=True)
        if r.status_code == 403 and self._refresh:
            self._url = self._refresh()
            r = self._requests.get(self._url, headers={"Range": "bytes=0-0"},
                                   timeout=self._timeout, stream=True)
        r.raise_for_status()
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return int(r.headers.get("Content-Length", 0))

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset, whence=io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size=-1) -> bytes:
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._pos + size, self._size) - 1
        if end < self._pos:
            return b""
        headers = {"Range": f"bytes={self._pos}-{end}"}
        r = self._requests.get(self._url, headers=headers, timeout=self._timeout, stream=True)
        if r.status_code == 403 and self._refresh:
            self._url = self._refresh()
            r = self._requests.get(self._url, headers=headers, timeout=self._timeout, stream=True)
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data


def extract_members(url, dest: Path, pattern: str, refresh=None) -> list[str]:
    """Extract every member of the remote zip whose name matches ``pattern``."""
    rx = re.compile(pattern)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    with zipfile.ZipFile(HttpRangeFile(url, refresh=refresh)) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not rx.search(member):
                continue
            zf.extract(member, dest)
            got.append(member)
            print(f"[range] extracted {member}")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernel", required=True, help="owner/slug")
    ap.add_argument("--zip", default="checkpoints.zip", help="output filename to read")
    ap.add_argument("--pattern", required=True, help="regex on member paths")
    ap.add_argument("--dest", default=".", help="extraction root (repo root)")
    args = ap.parse_args(argv)

    def refresh():
        return kernel_output_file_url(args.kernel, args.zip)

    try:
        url = refresh()
    except Exception as exc:  # noqa: BLE001
        print(f"[range] could not resolve URL: {exc}\n"
              "[range] fall back to: kaggle kernels output "
              f"{args.kernel} --file-pattern '^{args.zip}$'")
        return 2
    try:
        got = extract_members(url, Path(args.dest), args.pattern, refresh=refresh)
    except Exception as exc:  # noqa: BLE001
        print(f"[range] Range read failed ({exc}); fall back to CLI or "
              "curl -L -C - --retry 30 (PLAN_PHASE3 1.8.3)")
        return 2
    print(f"[range] extracted {len(got)} member(s)")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
