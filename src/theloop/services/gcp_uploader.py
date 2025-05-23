import logging
from typing import AsyncGenerator

import aiohttp
from google.cloud import storage
from google.cloud.storage import Client
from tqdm.asyncio import tqdm

_logger = logging.getLogger(__name__)


class GcpUploader:
    def __init__(self, client: Client, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        self.client = client
        self.chunk_size = chunk_size
        _logger.info(f"GcpUploader initialized with chunk_size: {chunk_size}")

    async def upload_stream_async(
        self,
        url: str,
        bucket_name: str,
        file_path: str,
    ) -> None:
        _logger.info(
            f"Starting upload from URL: {url} to bucket: {bucket_name}, path: {file_path}"
        )
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        _logger.debug(f"Created blob reference for path: {file_path}")

        async with aiohttp.ClientSession() as session:
            _logger.debug(f"Created HTTP session, requesting URL: {url}")
            async with session.get(url) as response:
                _logger.debug(f"HTTP response status: {response.status}")
                response.raise_for_status()

                # Get content length for progress bar
                content_length = response.headers.get("content-length")
                total_size = int(content_length) if content_length else None

                if total_size:
                    _logger.info(f"File size: {total_size} bytes")
                    progress_bar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Uploading to {bucket_name}/{file_path}",
                    )
                else:
                    _logger.info(
                        "File size unknown, showing speed and progress"
                    )
                    progress_bar = tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Uploading to {bucket_name}/{file_path}",
                    )

                async def stream_generator() -> AsyncGenerator[bytes, None]:
                    chunk_count = 0
                    total_downloaded = 0
                    try:
                        async for chunk in response.content.iter_chunked(
                            self.chunk_size
                        ):
                            chunk_count += 1
                            chunk_size = len(chunk)
                            total_downloaded += chunk_size

                            # Update progress bar
                            progress_bar.update(chunk_size)

                            _logger.debug(
                                f"Processing chunk {chunk_count}, size: {chunk_size} bytes"
                            )
                            yield chunk

                        _logger.info(
                            f"Finished streaming, total chunks processed: {chunk_count}, total bytes: {total_downloaded}"
                        )
                    finally:
                        progress_bar.close()

                await self._upload_from_async_generator(
                    blob, stream_generator()
                )
                _logger.info(
                    f"Successfully uploaded to GCS: {bucket_name}/{file_path}"
                )

    async def _upload_from_async_generator(
        self, blob: storage.Blob, data_stream: AsyncGenerator[bytes, None]
    ) -> None:
        _logger.debug("Starting to collect chunks for upload")
        chunks = []
        async for chunk in data_stream:
            chunks.append(chunk)

        combined_data = b"".join(chunks)
        total_size = len(combined_data)
        _logger.info(f"Uploading {total_size} bytes to GCS blob: {blob.name}")
        blob.upload_from_string(combined_data)
        _logger.debug(f"Upload completed for blob: {blob.name}")
