## Hackathon IMDB Auto-Fill Streamlit App

This project is a Streamlit experience that turns product imagery into the hackathon Item Master Database (IMDB) submission format. It wraps product-evidence grouping, multimodal extraction, barcode scanning, workbook-style normalization, human review, duplicate handling, ground-truth evaluation, and CSV/Excel export into one workflow.

### Key Features

- **Content-based image grouping** for arbitrary upload filenames using barcode and catalog evidence instead of filename prefixes.
- **Provider toggle** for Cohere or OpenAI, with Cohere as the default vision provider.
- **Exact 13-column hackathon export** with `ITEM_NAME`, `BARCODE`, `MANUFACTURER`, `BRAND`, `WEIGHT`, `PACKAGING  TYPE`, `COUNTRY`, `VARIANT`, `TYPE`, `FRAGRANCE_FLAVOR`, `PROMOTION`, `ADDONS`, and `TAGLINE`.
- **Workbook-style normalization** for uppercase values, pack syntax, compact weights, canonical packaging, country aliases, and checksum-valid barcodes.
- **Confidence-aware review** with a configurable low-confidence threshold.
- **Image-backed field review** with thumbnails beside editable field cards.
- **Inline editing plus split/merge controls** for product groups that need manual row adjustment.
- **Duplicate suggestions** driven by barcode, brand, and weight heuristics.
- **Validation scorecard** for barcode checksum, pack parsing, required-field completion, duplicates, and aligned ground-truth matching.
- **Main-workflow export** to `predictions.csv` or `predictions.xlsx`, saved locally for download.

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
   | `VLM_PROVIDER` | Default provider in the sidebar: `cohere` or `openai` (defaults to `cohere`). |
   | `COHERE_API_KEY` | Cohere API key for the default vision model. |
   | `COHERE_MODEL` | Cohere model name (defaults to `command-a-vision-07-2025`). |
   | `OPENAI_API_KEY` | Optional OpenAI API key for the fallback provider. Existing `VLM_API_KEY` still works as an alias. |
   | `OPENAI_MODEL` | Optional OpenAI model name. Existing `VLM_MODEL` still works as an alias. |
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

4. Upload arbitrary product photos on the first screen and click **Identify product groups**. The app groups images by product evidence before extraction, regardless of where the files came from.

### Testing

Run the unit suite once dependencies are installed:

```bash
pytest
```

Tests focus on the data contract, product-evidence grouping, validation, pack parsing, normalization, evaluation, exporter, pipeline, and duplicate heuristics. They skip external API calls by stubbing the vision client.

### Visual QA

Before recording or presenting, verify the app in Chrome at both desktop and mobile sizes:

- Initial load shows the five-step workflow with upload-first entry into grouping and extraction.
- Arbitrary uploads show an Identify Product Groups step and reviewable candidate clusters before extraction.
- Extracted uploads show product thumbnails, editable field cards, validation scorecard, duplicate status, and export preview.
- A manual field edit persists and updates validation.
- CSV or Excel generation exposes a visible download button in the main Export step.
- Mobile width around `390 x 844` has no horizontal overflow and no required action hidden only in the sidebar.

### Project Layout

- `app.py` – Streamlit entrypoint.
- `imdb_app/` – Reusable core modules (config, pipeline, storage, normalization).
- `tests/` – Pytest-based verification of core behaviors.
- `data/` – Sample imagery and seed lookup tables.
- `docs/` – Prompt iterations and demo scripts.

### Notes

- Missing provider config, HTTP/API failures, and response-parse failures now surface as explicit extraction errors instead of producing empty rows.
- The pipeline gracefully skips barcode extraction when PyZbar or Pillow are unavailable.
- For offline environments, pre-build wheels for pandas/openpyxl or install once with network access, then reuse the virtual environment.
