"""Live read tests against a public Cloudflare R2 bucket over HTTP(S).

The rest of the HTTP(S) tests are offline: they assert on `example.com` URLs and
never issue a request. These tests exercise the same pathlib and filesystem
extensions against a real object-storage endpoint, which is the case they were
added for, and pin down the behaviours documented in `alp_data.io.filesystem`
and `docs/io.md` (no listing API, range reads, GET-based `exists`).

They are marked `network` and skip when the bucket is unreachable, so an offline
checkout still passes. Run only these with `pytest -m network`, or skip them
with `pytest -m "not network"`.
"""

import json

import pytest
from fsspec.implementations.http import HTTPFileSystem

from alp_data.io import anypath, exists, filesystem_from_path, read_text
from alp_data.io.paths import PureHTTPPath, PureHTTPSPath

R2_PUBLIC_DIR = "https://pub-ad974658c4664976b92e83cd9e627e83.r2.dev/test"
JSONL_NAME = "beans_v0.1.0_raw_speech_commands_val_v2.jsonl"
JSONL_URL = f"{R2_PUBLIC_DIR}/{JSONL_NAME}"
MISSING_URL = f"{R2_PUBLIC_DIR}/this-object-does-not-exist.jsonl"

JSONL_SIZE = 1394890
"""Size in bytes of `JSONL_URL`, used to check reads are complete."""

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", autouse=True)
def require_public_bucket():
    """Skip the whole module when the public bucket cannot be reached."""
    try:
        HTTPFileSystem().info(JSONL_URL)
    except Exception as exc:  # noqa: BLE001 - any failure means "no bucket, no test"
        pytest.skip(f"public R2 bucket is unreachable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def jsonl_text():
    """The full contents of `JSONL_URL`, fetched once for the module."""
    return read_text(JSONL_URL)


def test_filesystem_from_public_url():
    """A public R2 URL resolves to an `HTTPFileSystem`, not the S3 backend."""
    assert isinstance(filesystem_from_path(JSONL_URL), HTTPFileSystem)


def test_anypath_parses_public_url():
    """The URL round-trips through `anypath` with the filename parsed out of it."""
    path = anypath(JSONL_URL)
    assert isinstance(path, PureHTTPSPath)
    assert str(path) == JSONL_URL
    assert path.name == JSONL_NAME
    assert path.suffix == ".jsonl"
    assert path.bucket == "pub-ad974658c4664976b92e83cd9e627e83.r2.dev"


def test_read_text_returns_whole_object(jsonl_text):
    """`read_text` downloads the complete object, not a truncated first chunk."""
    assert len(jsonl_text) == JSONL_SIZE
    assert jsonl_text.endswith("\n")


def test_read_text_content_is_valid_jsonl(jsonl_text):
    """Every line of the fetched manifest parses as JSON."""
    lines = jsonl_text.splitlines()
    assert len(lines) == 9981

    first = json.loads(lines[0])
    assert set(first) == {"label", "file_name", "local_path", "labels_as_list"}
    assert first["file_name"] == "067f61e2_nohash_0.wav"

    assert all(json.loads(line) for line in lines)


def test_read_text_from_joined_path(jsonl_text):
    """A path built by joining, rather than a literal URL, reads the same object."""
    path = anypath(R2_PUBLIC_DIR) / JSONL_NAME
    assert isinstance(path, PureHTTPSPath)
    assert str(path) == JSONL_URL
    assert read_text(path) == jsonl_text


def test_read_text_over_plain_http(jsonl_text):
    """A `http://` URL reads the object too; the endpoint redirects to HTTPS."""
    http_url = JSONL_URL.replace("https://", "http://", 1)
    assert isinstance(anypath(http_url), PureHTTPPath)
    assert isinstance(filesystem_from_path(http_url), HTTPFileSystem)
    assert read_text(http_url) == jsonl_text


def test_read_text_ignores_query_string(jsonl_text):
    """A query string is sent to the server but kept out of the parsed filename."""
    url = f"{JSONL_URL}?cachebust=1"
    path = anypath(url)
    assert path.name == JSONL_NAME
    assert path.suffix == ".jsonl"
    assert read_text(url) == jsonl_text


def test_info_reports_size_and_file_type():
    """`info` reads the object's metadata from the response headers."""
    info = filesystem_from_path(JSONL_URL).info(JSONL_URL)
    assert info["size"] == JSONL_SIZE
    assert info["type"] == "file"


def test_range_read_fetches_only_a_prefix(jsonl_text):
    """The endpoint honours range requests, so a partial read stays partial."""
    head = filesystem_from_path(JSONL_URL).cat_file(JSONL_URL, start=0, end=32)
    assert head == jsonl_text[:32].encode()


def test_seek_reads_from_the_middle_of_the_object(jsonl_text):
    """A seek followed by a read lands at the requested byte offset."""
    with filesystem_from_path(JSONL_URL).open(JSONL_URL, "rb") as f:
        f.seek(1000)
        chunk = f.read(16)
    assert chunk == jsonl_text[1000:1016].encode()


def test_exists_distinguishes_present_and_missing_objects():
    """A missing key answers 404, which `exists` reports as False."""
    assert exists(JSONL_URL) is True
    assert exists(MISSING_URL) is False


def test_ls_is_unavailable_on_an_object_storage_endpoint():
    """There is no index document to scrape, so listing fails as documented."""
    with pytest.raises(FileNotFoundError):
        filesystem_from_path(R2_PUBLIC_DIR).ls(R2_PUBLIC_DIR)


def test_glob_returns_nothing_instead_of_raising():
    """`glob` silently finds no files here — hence the "use a manifest" guidance."""
    assert filesystem_from_path(R2_PUBLIC_DIR).glob(f"{R2_PUBLIC_DIR}/*.jsonl") == []
