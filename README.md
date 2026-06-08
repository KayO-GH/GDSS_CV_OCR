## IMDB Auto-Fill Streamlit App

This project is a single Streamlit experience that turns product imagery into structured Item Master data (IMDB). It wraps a multimodal extraction pipeline, lightweight normalization, duplicate detection, and CSV/Excel export into one interactive workflow.

### Key Features

- **Batch image upload** backed by a vision-language model and optional barcode scan.
- **Confidence-aware review** with a configurable low-confidence threshold.
- **Inline editing** that records manual overrides for auditing.
- **Duplicate suggestions** driven by barcode, brand, and weight heuristics.
- **One-click export** to CSV or Excel, saved locally for download.

### Prerequisites

- Python 3.11 or newer.
- [`uv`](https://github.com/astral-sh/uv) or `pip` for dependency management.
- Optional system libraries for Pillow/PyZbar (e.g., `libjpeg`, `zbar`) if you want barcode decoding locally.

### Setup

1. Copy the example environment file and provide your model credentials:

   ```bash
   cp .env.example .env
   ```

   | Variable | Description |
   | --- | --- |
   | `VLM_API_KEY` | API key for the multimodal model (e.g., OpenAI Responses API). |
   | `VLM_API_URL` | Optional override for the responses endpoint. |
   | `VLM_MODEL` | Model name to call (defaults to `gpt-4o-mini`). |
   | `REQUEST_TIMEOUT_SECONDS` | Timeout for outbound model requests. |
   | `CONFIDENCE_THRESHOLD` | Initial threshold for highlighting low-confidence fields. |

2. Install dependencies:

   ```bash
   uv sync  # or: pip install -e .[dev]
   ```

3. Launch the Streamlit interface:

   ```bash
   uv run streamlit run app.py
   # or, once dependencies are installed: streamlit run app.py
   ```

### Testing

Run the lightweight unit suite once dependencies are installed:

```bash
pytest
```

Tests focus on the data pipeline, exporter, and duplicate heuristics. They skip external API calls by stubbing the vision client.

### Project Layout

- `app.py` – Streamlit entrypoint.
- `imdb_app/` – Reusable core modules (config, pipeline, storage, normalization).
- `tests/` – Pytest-based verification of core behaviors.
- `data/` – Sample imagery and seed lookup tables.
- `docs/` – Prompt iterations and demo scripts.

### Notes

- Without a `VLM_API_KEY`, the app still runs but returns empty fields to let you explore the UI.
- The pipeline gracefully skips barcode extraction when PyZbar or Pillow are unavailable.
- For offline environments, pre-build wheels for pandas/openpyxl or install once with network access, then reuse the virtual environment.
