import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from google.api_core import exceptions as gcp_exceptions
from google.cloud.storage import Blob, Bucket, Client

from theloop.services.gcp_uploader import GcpUploader


class TestGcpUploader:
    """Test suite for GcpUploader class."""

    def test_init_default_chunk_size(self):
        """Test initialization with default chunk size."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)

        assert uploader.client == mock_client
        assert uploader.chunk_size == 1024 * 1024  # 1MB default

    def test_init_custom_chunk_size(self):
        """Test initialization with custom chunk size."""
        mock_client = Mock(spec=Client)
        custom_chunk_size = 512 * 1024  # 512KB
        uploader = GcpUploader(mock_client, custom_chunk_size)

        assert uploader.client == mock_client
        assert uploader.chunk_size == custom_chunk_size

    def test_init_zero_chunk_size(self):
        """Test initialization with zero chunk size raises ValueError."""
        mock_client = Mock(spec=Client)
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            GcpUploader(mock_client, 0)

    def test_init_negative_chunk_size(self):
        """Test initialization with negative chunk size raises ValueError."""
        mock_client = Mock(spec=Client)
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            GcpUploader(mock_client, -1024)

    @pytest.mark.asyncio
    async def test_upload_stream_async_success(self):
        """Test successful upload of a small file."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client, chunk_size=1024)

        with patch("aiohttp.ClientSession") as mock_session_class:
            # Create a proper mock session and response
            mock_session = Mock()
            mock_response = Mock()

            # Mock the async context managers
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__.return_value = mock_response
            mock_response_cm.__aexit__.return_value = None
            mock_session.get.return_value = mock_response_cm

            # Mock response methods and content
            mock_response.raise_for_status = Mock()

            # Mock the async iteration over chunks
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

        # Verify calls
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("path/to/file.txt")
        mock_session.get.assert_called_once_with("https://example.com/file.txt")
        mock_response.raise_for_status.assert_called_once()
        mock_response.content.iter_chunked.assert_called_once_with(1024)
        mock_blob.upload_from_string.assert_called_once_with(b"test data chunk")

    @pytest.mark.asyncio
    async def test_upload_stream_async_multiple_chunks(self):
        """Test upload with multiple chunks."""
        mock_client = Mock(spec=Client)
        mock_bucket = Mock(spec=Bucket)
        mock_blob = Mock(spec=Blob)

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GcpUploader(mock_client, chunk_size=10)

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

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
    async def test_upload_stream_async_http_404_error(self):
        """Test upload failure due to HTTP 404 error."""
        mock_client = Mock(spec=Client)
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
                    request_info=Mock(), history=(), status=404
                )
            )

            with pytest.raises(aiohttp.ClientResponseError):
                await uploader.upload_stream_async(
                    "https://example.com/not-found.txt",
                    "test-bucket",
                    "not-found.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_http_500_error(self):
        """Test upload failure due to HTTP 500 error."""
        mock_client = Mock(spec=Client)
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
                    request_info=Mock(), history=(), status=500
                )
            )

            with pytest.raises(aiohttp.ClientResponseError):
                await uploader.upload_stream_async(
                    "https://example.com/server-error.txt",
                    "test-bucket",
                    "server-error.txt",
                )

    @pytest.mark.asyncio
    async def test_upload_stream_async_connection_error(self):
        """Test upload failure due to connection error."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_session.get.side_effect = aiohttp.ClientConnectorError(
                connection_key=Mock(), os_error=OSError("Connection failed")
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

            mock_session.get.side_effect = asyncio.TimeoutError(
                "Request timeout"
            )

            with pytest.raises(asyncio.TimeoutError):
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
        assert len(expected_data) == 100 * 1024  # 100KB

    @pytest.mark.asyncio
    async def test_upload_stream_async_special_characters_in_path(self):
        """Test upload with special characters in bucket and file path."""
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

        mock_client.bucket.assert_called_once_with("test-bucket-with-dashes")
        mock_bucket.blob.assert_called_once_with(
            "path/with spaces/file & symbols!.txt"
        )

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_success(self):
        """Test _upload_from_async_generator with successful upload."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)
        uploader = GcpUploader(mock_client)

        async def test_generator() -> AsyncGenerator[bytes, None]:
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        await uploader._upload_from_async_generator(mock_blob, test_generator())

        mock_blob.upload_from_string.assert_called_once_with(
            b"chunk1chunk2chunk3"
        )

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_empty(self):
        """Test _upload_from_async_generator with empty generator."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)
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
        mock_blob.upload_from_string.side_effect = (
            gcp_exceptions.GoogleAPIError("Upload failed")
        )

        uploader = GcpUploader(mock_client)

        async def test_generator() -> AsyncGenerator[bytes, None]:
            yield b"test data"

        with pytest.raises(gcp_exceptions.GoogleAPIError):
            await uploader._upload_from_async_generator(
                mock_blob, test_generator()
            )

    @pytest.mark.asyncio
    async def test_upload_from_async_generator_large_chunks(self):
        """Test _upload_from_async_generator with very large chunks."""
        mock_client = Mock(spec=Client)
        mock_blob = Mock(spec=Blob)
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
        assert len(expected_data) == 2 * 1024 * 1024  # 2MB

    @pytest.mark.asyncio
    async def test_upload_stream_async_invalid_url_format(self):
        """Test upload with invalid URL format."""
        mock_client = Mock(spec=Client)
        uploader = GcpUploader(mock_client)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            mock_session.get.side_effect = aiohttp.InvalidURL("Invalid URL")

            with pytest.raises(aiohttp.InvalidURL):
                await uploader.upload_stream_async(
                    "not-a-valid-url", "test-bucket", "file.txt"
                )

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

    def test_init_invalid_chunk_size_zero(self):
        """Test that chunk_size of 0 raises ValueError."""
        mock_client = Mock(spec=Client)
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            GcpUploader(mock_client, chunk_size=0)

    def test_init_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises ValueError."""
        mock_client = Mock(spec=Client)
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            GcpUploader(mock_client, chunk_size=-100)
