# Demo Script: IMDB Auto-Fill Prototype

## Overview
This script narrates the end-to-end workflow for stakeholders. Target duration: 3–4 minutes.

1. **Intro (20s)**
   - Explain the manual IMDB pain point and the goal: image → auto-filled master data.
   - Mention that the prototype supports live Cohere/OpenAI extraction plus a curated offline demo path.

2. **Add Images (35s)**
   - Click **Use curated demo data**.
   - Show BAMA, TAPOK, and ZESTA rows loading from workbook-backed fixtures.
   - Point out that live sample extraction and arbitrary uploads remain available.
   - For arbitrary uploads, explain that the app identifies product groups from barcode and label evidence rather than file names.

3. **Review Fields (70s)**
   - Show product thumbnails beside the editable field cards.
   - Open one required field and make a small edit.
   - Point out confidence/source notes and the 13-column workbook mapping.

4. **Validate & Dedupe (45s)**
   - Show barcode checksum status, pack parsing, required completion, and duplicate status.
   - For TAPOK or ZESTA, highlight that pack syntax is interpreted separately from the raw product name.

5. **Export (45s)**
   - Use the output filters to demonstrate search-ready data.
   - Generate CSV or Excel and show the visible download button in the main Export step.

6. **Wrap-Up (20s)**
   - Summarize accuracy benefits, human-in-loop edits, and integration next steps.
   - End on the validation scorecard.
