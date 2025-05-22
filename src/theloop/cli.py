import asyncio
from pathlib import Path
from typing import Optional

import typer
from google.cloud import storage
from google.oauth2 import service_account

from theloop.interfaces import Uploader
from theloop.services.config_manager import ConfigManager, Settings
from theloop.services.gcp_uploader import GcpUploader
from theloop.services.logging_configurator import LoggingConfigurator

app = typer.Typer(
    name="theloop",
    help="Upload files from URLs to cloud storage",
    add_completion=False,
)

config_manager = ConfigManager()


@app.command()
def upload(
    url: str = typer.Argument(..., help="URL of the file to upload"),
    bucket: str = typer.Argument(..., help="Target bucket name"),
    path: str = typer.Argument(..., help="Destination file path in bucket"),
    provider: str = typer.Option(
        "gcp", "--provider", "-p", help="Cloud provider (gcp)"
    ),
    credentials_file: Optional[str] = typer.Option(
        None, "--credentials", "-c", help="Path to service account JSON file"
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project", help="GCP project ID (overrides default)"
    ),
) -> None:
    """Upload a file from URL to cloud storage."""
    asyncio.run(
        upload_async(url, bucket, path, provider, credentials_file, project_id)
    )


@app.command()
def settings() -> None:
    settings = config_manager.load_config()
    print(settings.model_dump_json(indent=2))


async def upload_async(
    url: str,
    bucket: str,
    path: str,
    provider: str,
    credentials_file: Optional[str],
    project_id: Optional[str],
) -> None:
    settings = config_manager.load_config()

    logging_configurator = LoggingConfigurator(settings)
    logging_configurator.setup_logging()

    uploader = _get_uploader(provider, settings, credentials_file, project_id)

    try:
        await uploader.upload_stream_async(url, bucket, path)
        typer.echo(
            f"✅ Successfully uploaded {url} to {provider}://{bucket}/{path}"
        )
    except Exception as e:
        typer.echo(f"❌ Upload failed: {e}", err=True)
        raise typer.Exit(1)


def _get_uploader(
    provider: str,
    settings: Settings,
    credentials_file: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Uploader:
    if provider == "gcp":
        client = _create_gcp_client(credentials_file, project_id)
        return GcpUploader(client, settings.chunk_size)
    else:
        typer.echo(f"❌ Unsupported provider: {provider}", err=True)
        raise typer.Exit(1)


def _create_gcp_client(
    credentials_file: Optional[str] = None,
    project_id: Optional[str] = None,
) -> storage.Client:
    try:
        if credentials_file:
            credentials_path = Path(credentials_file).expanduser()
            if not credentials_path.exists():
                typer.echo(
                    f"❌ Credentials file not found: {credentials_file}",
                    err=True,
                )
                raise typer.Exit(1)

            credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path)
            )
            return storage.Client(credentials=credentials, project=project_id)
        else:
            return storage.Client(project=project_id)

    except Exception as e:
        typer.echo(f"❌ Failed to create GCP client: {e}", err=True)
        raise typer.Exit(1)
