from typing import AsyncGenerator

import aiohttp
from google.cloud import storage
from google.cloud.storage import Client


class GcpUploader:
    def __init__(self, client: Client, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        self.client = client
        self.chunk_size = chunk_size

    async def upload_stream_async(
        self,
        url: str,
        bucket_name: str,
        file_path: str,
    ) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()

                async def stream_generator() -> AsyncGenerator[bytes, None]:
                    async for chunk in response.content.iter_chunked(
                        self.chunk_size
                    ):
                        yield chunk

                await self._upload_from_async_generator(
                    blob, stream_generator()
                )

    async def _upload_from_async_generator(
        self, blob: storage.Blob, data_stream: AsyncGenerator[bytes, None]
    ) -> None:
        chunks = []
        async for chunk in data_stream:
            chunks.append(chunk)

        combined_data = b"".join(chunks)
        blob.upload_from_string(combined_data)
