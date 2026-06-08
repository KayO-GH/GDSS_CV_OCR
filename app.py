"""Streamlit application for IMDB Auto-Fill."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING

import pandas as pd
import streamlit as st

from imdb_app import IMDB_ATTRIBUTES, ProductRecord, settings
from imdb_app.exporter import Exporter
from imdb_app.pipeline import ExtractionPipeline, get_pipeline, run_pipeline_sync
from imdb_app.store import ProductStore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(page_title="IMDB Auto-Fill", layout="wide")


def get_store() -> ProductStore:
    if "store" not in st.session_state:
        st.session_state.store = ProductStore()
    return st.session_state.store


def get_pipeline_instance() -> ExtractionPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()
    return st.session_state.pipeline


def get_suggestions() -> List[dict]:
    return st.session_state.setdefault("suggestions", [])


def set_suggestions(suggestions: List[dict]) -> None:
    st.session_state.suggestions = suggestions


def process_uploads(files: Iterable["UploadedFile"], pipeline: ExtractionPipeline, store: ProductStore) -> tuple[list[ProductRecord], list[str]]:
    processed: list[ProductRecord] = []
    errors: list[str] = []

    if not files:
        return processed, errors

    status = st.status("Processing images...", expanded=True)
    try:
        for upload in files:
            filename = upload.name
            status.write(f"Extracting data from {filename}")
            image_bytes = upload.getvalue()
            if not image_bytes:
                errors.append(f"{filename}: empty file")
                continue
            try:
                record = run_pipeline_sync(pipeline, image_bytes, filename=filename)
            except Exception as exc:  # pragma: no cover - defensive logging for user feedback
                errors.append(f"{filename}: {exc}")
            else:
                store.upsert(record)
                processed.append(record)
        status.update(label="Extraction complete", state="complete")
    finally:
        status.stop()

    return processed, errors


def build_summary_frame(records: Iterable[ProductRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {"Record": format_record_name(record), "File": record.filename or "—"}
        for attr in IMDB_ATTRIBUTES:
            row[f"{attr} (value)"] = getattr(record, attr).value
            row[f"{attr} (confidence)"] = getattr(record, attr).confidence
        rows.append(row)
    return pd.DataFrame(rows)


def format_record_name(record: ProductRecord) -> str:
    return record.product_name.value or record.brand.value or record.filename or record.id[:8]


def render_record_editor(record: ProductRecord, threshold: float) -> None:
    title = format_record_name(record)
    with st.expander(title, expanded=False):
        st.caption(f"Record ID: {record.id}")
        for attr in IMDB_ATTRIBUTES:
            attribute = getattr(record, attr)
            label = attr.replace("_", " ").title()
            value = attribute.value or ""
            key = f"record-{record.id}-{attr}"
            new_value = st.text_input(label, value=value, key=key)
            cleaned_value = new_value.strip() or None
            if cleaned_value != attribute.value:
                attribute.value = cleaned_value
                if cleaned_value:
                    attribute.source = "manual_edit"
                    attribute.confidence = 1.0
                record.metadata.setdefault("edited", {})[attr] = True

            confidence = attribute.confidence or 0.0
            info_parts = [f"Confidence {confidence:.0%}"]
            if attribute.source:
                info_parts.append(f"Source: {attribute.source}")
            if attribute.notes:
                info_parts.append(f"Notes: {attribute.notes}")

            message = " • ".join(info_parts)
            if confidence < threshold:
                st.markdown(f":orange[{message or 'No confidence signal'}]")
            else:
                st.caption(message or "No confidence signal")


def render_merge_suggestions(records: List[ProductRecord], suggestions: List[dict]) -> None:
    if not suggestions:
        return

    id_lookup = {record.id: record for record in records}
    st.subheader("Possible Duplicates")
    for suggestion in suggestions:
        target = id_lookup.get(suggestion.get("record_id"))
        target_name = format_record_name(target) if target else suggestion.get("record_id")
        st.markdown(f"**{target_name}**")
        for candidate in suggestion.get("candidates", []):
            candidate_record = id_lookup.get(candidate.get("candidate_id"))
            candidate_name = format_record_name(candidate_record) if candidate_record else candidate.get("candidate_id")
            reasons = ", ".join(candidate.get("reasons", [])) or "No specific reasons"
            st.write(f"- {candidate_name} · score {candidate.get('score', 0):.2f} · {reasons}")


def render_export_controls(records: List[ProductRecord], exporter: Exporter) -> None:
    st.sidebar.subheader("Export")
    format_label = st.sidebar.radio("Format", options=["csv", "excel"], index=0, horizontal=True)
    export_disabled = not records

    if st.sidebar.button("Generate export", disabled=export_disabled):
        path = exporter.export(records, format=format_label)
        st.session_state.last_export_path = str(path)
        st.toast(f"Exported {path.name}")

    export_path_str = st.session_state.get("last_export_path")
    if export_path_str:
        export_path = Path(export_path_str)
        if export_path.exists():
            mime = "text/csv" if export_path.suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            st.sidebar.download_button(
                label=f"Download {export_path.name}",
                data=export_path.read_bytes(),
                file_name=export_path.name,
                mime=mime,
            )


def main() -> None:
    st.title("IMDB Auto-Fill Streamlit App")
    st.caption("Upload product imagery, auto-fill catalog attributes, and export curated data.")

    store = get_store()
    pipeline = get_pipeline_instance()
    exporter = st.session_state.setdefault("exporter", Exporter())

    st.sidebar.subheader("Configuration")
    threshold = st.sidebar.slider(
        "Low confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.default_confidence_threshold,
        step=0.05,
    )

    if settings.vlm_api_key:
        st.sidebar.success("VLM API key detected")
    else:
        st.sidebar.warning("No VLM API key provided. Fallback attributes will be empty.")

    if st.sidebar.button("Clear workspace", type="secondary", use_container_width=True):
        store.clear()
        st.session_state.store = ProductStore()
        st.session_state.pop("suggestions", None)
        st.session_state.pop("last_export_path", None)
        st.experimental_rerun()

    uploaded_files = st.file_uploader(
        "Upload product images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="uploader"
    )

    trigger_extraction = st.button("Run extraction", disabled=not uploaded_files)
    if trigger_extraction and uploaded_files:
        processed, errors = process_uploads(uploaded_files, pipeline, store)
        if processed:
            st.success(f"Processed {len(processed)} file(s).")
        if errors:
            error_buffer = io.StringIO()
            for item in errors:
                error_buffer.write(f"• {item}\n")
            st.error(error_buffer.getvalue())
        set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()]))
        st.session_state.uploader = None

    records = store.all()

    st.sidebar.button(
        "Recompute duplicates",
        disabled=not records,
        on_click=lambda: set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()])),
    )

    render_export_controls(records, exporter)

    if records:
        frame = build_summary_frame(records)
        st.subheader("Current Records")
        st.dataframe(frame, use_container_width=True, hide_index=True)

        st.subheader("Review & Edit")
        for record in records:
            render_record_editor(record, threshold)

        render_merge_suggestions(records, get_suggestions())
    else:
        st.info("Upload imagery to begin the extraction pipeline.")


if __name__ == "__main__":
    main()
