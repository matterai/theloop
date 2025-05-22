from typing import Protocol


class Uploader(Protocol):
    async def upload_stream_async(
        self,
        url: str,
        bucket_name: str,
        file_path: str,
    ) -> None:
        pass
