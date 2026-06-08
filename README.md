## Hackathon IMDB Auto-Fill Streamlit App

This project is a Streamlit experience that turns grouped product imagery into the hackathon Item Master Database (IMDB) submission format. It wraps multimodal extraction, barcode scanning, workbook-style normalization, human review, duplicate handling, ground-truth evaluation, and CSV/Excel export into one workflow.

### Key Features

- **Grouped image upload** backed by a vision-language model and optional barcode scan.
- **Exact 13-column hackathon export** with `ITEM_NAME`, `BARCODE`, `MANUFACTURER`, `BRAND`, `WEIGHT`, `PACKAGING  TYPE`, `COUNTRY`, `VARIANT`, `TYPE`, `FRAGRANCE_FLAVOR`, `PROMOTION`, `ADDONS`, and `TAGLINE`.
- **Workbook-style normalization** for uppercase values, compact weights, canonical packaging, country aliases, and digit-only barcodes.
- **Confidence-aware review** with a configurable low-confidence threshold.
- **Inline editing plus split/merge controls** for product groups that need manual row adjustment.
- **Duplicate suggestions** driven by barcode, brand, and weight heuristics.
- **Ground-truth evaluation** against `hackathon_material/Hackathon Materials/output_results.xlsx`.
- **One-click export** to `predictions.csv` or `predictions.xlsx`, saved locally for download.

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

Run the unit suite once dependencies are installed:

```bash
pytest
```

Tests focus on the data contract, grouping, normalization, evaluation, exporter, pipeline, and duplicate heuristics. They skip external API calls by stubbing the vision client.

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
