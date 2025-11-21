# images-mcp

Python package and MCP server for image conversion, PDF merging, text extraction, and image generation.

## Installation

Using Poetry:

```bash
poetry install
```

Or install from source with pip:

```bash
pip install .
```

This package depends on:
- Pillow (for image processing)
- `pillow-avif-plugin` (for AVIF support)
- `mcp` (for MCP server functionality)
- `unstructured` (for text extraction from documents)
- `openai` (for image generation)

## Usage

Python API:

```python
from images_mcp import convert_image, images_to_pdf, extract_text_from_file

# Convert PNG to JPEG at quality 85
output = convert_image("input.png", format="jpeg", quality=85)
print(output)

# Merge multiple images into a PDF
images_to_pdf(["img1.png", "img2.jpg"], "output.pdf", overwrite=True)

# Extract text from a document or image
text = extract_text_from_file("document.pdf")
print(text)
```

## Project layout

```
images-mcp/
├─ images_mcp/
│  ├─ __init__.py
│  └─ mcp_server.py
├─ pyproject.toml
├─ README.md
└─ .gitignore
```

## MCP server

This project includes an MCP server that exposes image processing tools for clients like Claude for Desktop.

### Available Tools

- **`convert`**: Convert images between formats (JPEG, PNG, WebP, AVIF, etc.)
- **`merge_pdf`**: Merge multiple images into a single PDF
- **`extract_text`**: Extract text from images, PDFs, and documents using Unstructured
- **`generate_image`**: Generate or edit images using OpenAI's image generation API

### Requirements

- Python 3.10+
- Poetry
- OpenAI API key (for `generate_image` tool - set in `.env` file as `OPENAI_API_KEY`)

### Install dependencies

```bash
poetry install
```

### Run the server (manual)

```bash
poetry run images-mcp
```

This starts an MCP stdio server.

### Configure Claude for Desktop (Windows)

Create or edit your Claude config at:

`%APPDATA%/Claude/claude_desktop_config.json`

Add the server configuration, replacing the path with your absolute project path. Use double backslashes or forward slashes in JSON paths:

```json
{
  "mcpServers": {
    "images-mcp": {
      "command": "poetry",
      "args": [
        "run",
        "images-mcp"
      ],
      "env": {
        "PYTHONUTF8": "1"
      },
      "workingDirectory": "C:/ABSOLUTE/PATH/TO/images-mcp"
    }
  }
}
```

Notes:
- The server uses stdio; avoid printing to stdout in your own changes.
- Ensure the `workingDirectory` points to the project root so Poetry resolves the environment.
- For the `generate_image` tool, create a `.env` file in the project root with your `OPENAI_API_KEY`.

After saving the config, restart Claude for Desktop. You should see tools named `convert`, `merge_pdf`, `extract_text`, and `generate_image` under the `images-mcp` server.

## License

MIT