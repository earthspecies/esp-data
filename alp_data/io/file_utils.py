from typing import Any

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT, PureHTTPPath, PureHTTPSPath, anypath


def exists(path: str | AnyPathT) -> bool:
    """Check if a file or directory exists.

    Parameters
    ----------
    path: str | AnyPathT
        File or directory to check.

    Returns
    -------
    bool
        True if the file or directory exists, False otherwise.

    Notes
    -----
    For HTTP(S) paths the check is a GET request, not a HEAD: `fsspec`'s
    `HTTPFileSystem` treats any response status below 400 as "exists". The
    response body is discarded, but the request is still more expensive than the
    metadata lookup used for `gs://` and `r2://` paths, so avoid calling this in
    a tight loop over HTTP(S) URLs. A server that answers a missing object with
    a 200 "not found" page will also report as existing.
    """
    return filesystem_from_path(path).exists(str(path))


def rm(
    path: str | AnyPathT,
    recursive: bool = False,
    maxdepth: int | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """Delete files.

    Parameters
    ----------
    path: str | AnyPathT
        File(s) to delete.
    recursive: bool
        If file(s) are directories, recursively delete contents and then also remove the
        directory
    maxdepth: int | None
        Depth to pass to walk for finding files to delete, if recursive. If None, there
        will be no limit and infinite recursion may be possible.

    Raises
    ------
    NotImplementedError
        If `path` is an HTTP(S) URL. HTTP(S) endpoints are read-only in `alp_data`.

    Notes
    -----
    HTTP(S) paths are rejected before any request is made. `fsspec`'s
    `HTTPFileSystem` does not implement file removal, so it would raise a bare
    `NotImplementedError` with no message anyway — and with `recursive=True` it
    would first issue several HTTP requests while expanding the path.
    """
    if isinstance(anypath(path), (PureHTTPSPath, PureHTTPPath)):
        raise NotImplementedError(
            f"Cannot delete over HTTP(S): {path}. HTTP(S) endpoints are read-only in "
            "alp_data; use a 'gs://' or 'r2://' path to delete an object."
        )

    filesystem_from_path(path).rm(str(path), recursive=recursive, maxdepth=maxdepth, **kwargs)
