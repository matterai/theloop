# TheLoop

A Python CLI tool for uploading files from URLs directly to cloud storage. Currently supports Google Cloud Storage (GCP) with a modular architecture designed for easy extension to other cloud providers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 Upload files directly from URLs to cloud storage
- ☁️ Google Cloud Storage support
- 📦 Streaming uploads with progress tracking
- ⚙️ Configurable settings and logging
- 🔧 Extensible architecture for additional cloud providers
- 🛡️ Async/await for efficient stream processing

## Installation

### Prerequisites

- Python 3.11 or higher
- Google Cloud Storage credentials (for GCP uploads)

### Install from PyPI

```bash
pip install theloop
```

### Install from Source

```bash
git clone https://github.com/matterai/theloop.git
cd theloop
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/matterai/theloop.git
cd theloop
uv pip install -e .
uv pip install pytest pytest-cov pytest-asyncio
```

## Usage

### Basic Upload

Upload a file from a URL to Google Cloud Storage:

```bash
theloop upload "https://example.com/file.pdf" "my-bucket" "uploads/file.pdf"
```

### Advanced Usage

#### Using Service Account Credentials

```bash
theloop upload \
  "https://example.com/file.pdf" \
  "my-bucket" \
  "uploads/file.pdf" \
  --credentials ~/path/to/service-account.json \
  --project my-gcp-project-id
```

#### Specify Cloud Provider

```bash
theloop upload \
  "https://example.com/file.pdf" \
  "my-bucket" \
  "uploads/file.pdf" \
  --provider gcp
```

### Configuration

View current settings:

```bash
theloop settings
```

Settings are stored in `~/.theloop/settings.json` and include:
- Logging configuration
- Chunk size for uploads
- Default cloud provider settings

### Authentication

#### Google Cloud Platform

1. **Application Default Credentials** (recommended for local development):
   ```bash
   gcloud auth application-default login
   ```

2. **Service Account Key**:
   ```bash
   theloop upload <url> <bucket> <path> --credentials /path/to/service-account.json
   ```

3. **Environment Variable**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   theloop upload <url> <bucket> <path>
   ```

## Development

### Project Structure

```
src/theloop/
├── __init__.py
├── cli.py                 # CLI interface using Typer
├── interfaces.py          # Protocol definitions
└── services/
    ├── config_manager.py  # Settings management
    ├── gcp_uploader.py    # GCP-specific upload logic
    └── logging_configurator.py
```

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/matterai/theloop.git
   cd theloop
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -e .
   uv pip install pytest pytest-cov pytest-asyncio
   ```

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Run from source**:
   ```bash
   uv run python -m src.main upload <url> <bucket> <path>
   ```

### Testing

The project uses pytest for testing with support for async code:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=theloop

# Run specific test file
pytest tests/test_uploader.py
```

Example test structure:
```python
import pytest

@pytest.mark.asyncio
async def test_async_upload():
    # Test async upload functionality
    pass
```

### Adding a New Cloud Provider

1. **Create uploader class** in `services/`:
   ```python
   from theloop.interfaces import Uploader

   class NewProviderUploader(Uploader):
       async def upload_stream_async(self, url: str, bucket: str, path: str) -> None:
           # Implement upload logic
           pass
   ```

2. **Update CLI** in `cli.py`:
   ```python
   def _get_uploader(provider: str, settings: Settings, ...) -> Uploader:
       if provider == "new-provider":
           return NewProviderUploader(...)
       # ... existing providers
   ```

### Code Style and Standards

- Follow PEP 8 for Python code style
- Use type hints throughout the codebase
- Implement proper error handling with user-friendly messages
- Use async/await for I/O operations
- Follow the existing patterns for configuration and logging

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the development guidelines
4. **Add tests** for new functionality
5. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```
6. **Commit your changes**:
   ```bash
   git commit -m "Add your feature description"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

### Contribution Guidelines

- Ensure all tests pass
- Add tests for new features
- Update documentation as needed
- Follow existing code style and patterns
- Write clear commit messages
- Keep pull requests focused and small

## License

This project is licensed under the MIT License - see the [LICENSE](#license-text) section below for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/matterai/theloop/issues)
- **Discussions**: [GitHub Discussions](https://github.com/matterai/theloop/discussions)

## Roadmap

- [ ] AWS S3 support
- [ ] Azure Blob Storage support
- [ ] Configuration file support
- [ ] Batch upload functionality
- [ ] Resume interrupted uploads

---

## License Text

MIT License

Copyright (c) 2024 Vladimir Vlasiuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
