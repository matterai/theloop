from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from google.cloud import exceptions as gcp_exceptions
from google.cloud.storage import Blob, Bucket, Client

from theloop.services.gcp_uploader import GcpUploader


class TestGcpUploader:
    def test_init_valid_chunk_size(self):
        """Test initialization with valid chunk size."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client, chunk_size=2048)
        assert uploader.chunk_size == 2048
        assert uploader.client == mock_client

    def test_init_default_chunk_size(self):
        """Test initialization with default chunk size."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)
        assert uploader.chunk_size == 1024 * 1024

    def test_init_invalid_chunk_size_zero(self):
        """Test initialization with zero chunk size raises ValueError."""
        mock_client = Mock(spec=Client)
        with pytest.raises(
            ValueError, match="chunk_size must be greater than 0"
        ):
            GcpUploader(mock_client, chunk_size=0)

    def test_init_invalid_chunk_size_negative(self):
        """Test initialization with negative chunk size raises ValueError."""
        mock_client = Mock(spec=Client)
        with pytest.raises(
            ValueError, match="chunk_size must be greater than 0"
        ):
            GcpUploader(mock_client, chunk_size=-1)

    @pytest.mark.asyncio
    async def test_upload_stream_async_success(self):
        """Test successful upload of a small file."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "path/to/file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client, chunk_size=1024)

        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "theloop.services.gcp_uploader.tqdm"
        ) as mock_tqdm:
            mock_session = Mock()
            mock_response = Mock()
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return string instead of Mock for content-length
            mock_response.headers = {"content-length": "16"}

            async def mock_iter_chunked(chunk_size):
                yield b"test data chunk"

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024)
            )

            await uploader.upload_stream_async(
                "https://example.com/file.txt",
                "test-bucket",
                "path/to/file.txt",
            )

            mock_blob.upload_from_string.assert_called_once_with(
                b"test data chunk"
            )

    @pytest.mark.asyncio
    async def test_upload_stream_async_multiple_chunks(self):
        """Test upload with multiple chunks."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "large-file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client, chunk_size=10)

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "theloop.services.gcp_uploader.tqdm"
        ) as mock_tqdm:
            mock_session = Mock()
            mock_response = Mock()
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return string instead of Mock for content-length
            mock_response.headers = {
                "content-length": str(sum(len(chunk) for chunk in chunks))
            }

            async def mock_iter_chunked(chunk_size):
                for chunk in chunks:
                    yield chunk

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(10)
            )

            await uploader.upload_stream_async(
                "https://example.com/large-file.txt",
                "test-bucket",
                "large-file.txt",
            )

            expected_data = b"".join(chunks)
            mock_blob.upload_from_string.assert_called_once_with(expected_data)

    @pytest.mark.asyncio
    async def test_upload_stream_async_empty_response(self):
        """Test upload with empty response."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "empty-file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "theloop.services.gcp_uploader.tqdm"
        ) as mock_tqdm:
            mock_session = Mock()
            mock_response = Mock()
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                return
                yield  # Never reached

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024 * 1024)
            )

            await uploader.upload_stream_async(
                "https://example.com/empty-file.txt",
                "test-bucket",
                "empty-file.txt",
            )

            mock_blob.upload_from_string.assert_called_once_with(b"")

    @pytest.mark.asyncio
    async def test_upload_stream_async_no_content_length_header(self):
        """Test upload when content-length header is missing."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "theloop.services.gcp_uploader.tqdm"
        ) as mock_tqdm:
            mock_session = Mock()
            mock_response = Mock()
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict without content-length
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                yield b"test data"

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024 * 1024)
            )

            await uploader.upload_stream_async(
                "https://example.com/file.txt",
                "test-bucket",
                "file.txt",
            )

            mock_blob.upload_from_string.assert_called_once_with(b"test data")

    @pytest.mark.asyncio
    async def test_upload_stream_async_http_error(self):
        """Test upload failure due to HTTP error."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status.side_effect = (
                aiohttp.ClientResponseError(
                    request_info=Mock(),
                    history=(),
                    status=404,
                    message="Not Found",
                )
            )

            with pytest.raises(aiohttp.ClientResponseError):
                await uploader.upload_stream_async(
                    "https://example.com/nonexistent.txt",
                    "test-bucket",
                    "nonexistent.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_network_error(self):
        """Test upload failure due to network error."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_session.get.side_effect = aiohttp.ClientConnectorError(
                connection_key=Mock(), os_error=OSError("Network error")
            )

            with pytest.raises(aiohttp.ClientConnectorError):
                await uploader.upload_stream_async(
                    "https://unreachable.example.com/file.txt",
                    "test-bucket",
                    "file.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_timeout_error(self):
        """Test upload failure due to timeout."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_session.get.side_effect = aiohttp.ServerTimeoutError(
                "Timeout error"
            )

            with pytest.raises(aiohttp.ServerTimeoutError):
                await uploader.upload_stream_async(
                    "https://slow.example.com/file.txt",
                    "test-bucket",
                    "file.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_gcp_bucket_not_found(self):
        """Test upload failure when GCP bucket doesn't exist."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_string.side_effect = gcp_exceptions.NotFound(
            "Bucket not found"
        )

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                yield b"test data"

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024 * 1024)
            )

            with pytest.raises(gcp_exceptions.NotFound):
                await uploader.upload_stream_async(
                    "https://example.com/file.txt",
                    "non-existent-bucket",
                    "file.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_gcp_permission_denied(self):
        """Test upload failure due to GCP permission denied."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "file.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_string.side_effect = gcp_exceptions.Forbidden(
            "Permission denied"
        )

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                yield b"test data"

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024 * 1024)
            )

            with pytest.raises(gcp_exceptions.Forbidden):
                await uploader.upload_stream_async(
                    "https://example.com/file.txt",
                    "forbidden-bucket",
                    "file.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_large_file(self):
        """Test upload of a large file with many chunks."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "large-file.bin"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client, chunk_size=1024)

        large_chunks = [b"x" * 1024 for _ in range(100)]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                for chunk in large_chunks:
                    yield chunk

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024)
            )

            await uploader.upload_stream_async(
                "https://example.com/large-file.bin",
                "test-bucket",
                "large-file.bin",
            )

            expected_data = b"".join(large_chunks)
            mock_blob.upload_from_string.assert_called_once_with(expected_data)

    @pytest.mark.asyncio
    async def test_upload_stream_async_special_characters_in_path(self):
        """Test upload with special characters in bucket and file path."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "path/with spaces/file & symbols!.txt"

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked(chunk_size):
                yield b"test data"

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked(1024 * 1024)
            )

            await uploader.upload_stream_async(
                "https://example.com/file with spaces & symbols!.txt",
                "test-bucket-with-dashes",
                "path/with spaces/file & symbols!.txt",
            )

            mock_blob.upload_from_string.assert_called_once_with(b"test data")

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_success(self):
        """Test _upload_from_async_generator with successful upload."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "test-blob"

        uploader = GcpUploader(mock_client)

        async def test_generator() -> AsyncGenerator[bytes, None]:
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        await uploader._upload_from_async_generator(mock_blob, test_generator())

        expected_data = b"chunk1chunk2chunk3"
        mock_blob.upload_from_string.assert_called_once_with(expected_data)

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_empty(self):
        """Test _upload_from_async_generator with empty generator."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "empty-blob"

        uploader = GcpUploader(mock_client)

        async def empty_generator() -> AsyncGenerator[bytes, None]:
            return
            yield  # This will never execute

        await uploader._upload_from_async_generator(
            mock_blob, empty_generator()
        )

        mock_blob.upload_from_string.assert_called_once_with(b"")

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_single_chunk(self):
        """Test _upload_from_async_generator with single chunk."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "single-chunk-blob"

        uploader = GcpUploader(mock_client)

        async def single_chunk_generator() -> AsyncGenerator[bytes, None]:
            yield b"single chunk data"

        await uploader._upload_from_async_generator(
            mock_blob, single_chunk_generator()
        )

        mock_blob.upload_from_string.assert_called_once_with(
            b"single chunk data"
        )

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_blob_upload_error(self):
        """Test _upload_from_async_generator when blob upload fails."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "error-blob"

        mock_blob.upload_from_string.side_effect = (
            gcp_exceptions.GoogleCloudError("Upload failed")
        )

        uploader = GcpUploader(mock_client)

        async def test_generator() -> AsyncGenerator[bytes, None]:
            yield b"test data"

        with pytest.raises(gcp_exceptions.GoogleCloudError):
            await uploader._upload_from_async_generator(
                mock_blob, test_generator()
            )

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_large_chunks(self):
        """Test _upload_from_async_generator with very large chunks."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)

        # Fix: Add name attribute to blob mock
        mock_blob.name = "large-chunks-blob"

        uploader = GcpUploader(mock_client)

        large_chunk_size = 1024 * 1024  # 1MB

        async def large_chunk_generator() -> AsyncGenerator[bytes, None]:
            yield b"A" * large_chunk_size
            yield b"B" * large_chunk_size

        await uploader._upload_from_async_generator(
            mock_blob, large_chunk_generator()
        )

        expected_data = b"A" * large_chunk_size + b"B" * large_chunk_size
        mock_blob.upload_from_string.assert_called_once_with(expected_data)

    @pytest.mark.asyncio
    async def test_upload_stream_async_content_encoding_error(self):
        """Test upload failure during content streaming."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            mock_response.raise_for_status = Mock()
            # Fix: Return proper headers dict
            mock_response.headers = {}

            async def mock_iter_chunked_with_error(chunk_size):
                raise aiohttp.ClientResponseError(
                    request_info=Mock(),
                    history=(),
                    status=400,
                    message="Content encoding error",
                )
                yield  # Never reached

            mock_response.content.iter_chunked = Mock(
                return_value=mock_iter_chunked_with_error(1024 * 1024)
            )

            with pytest.raises(aiohttp.ClientResponseError):
                await uploader.upload_stream_async(
                    "https://example.com/corrupted-file.txt",
                    "test-bucket",
                    "corrupted-file.txt",
                )
