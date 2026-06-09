"""Streamlit application for IMDB Auto-Fill."""

from __future__ import annotations

import asyncio
import copy
import io
import uuid
from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING

import pandas as pd
import streamlit as st

from imdb_app import EXPORT_COLUMNS, IMDB_ATTRIBUTES, ProductRecord, settings
from imdb_app.evaluator import GROUND_TRUTH_PATH, evaluate_records
from imdb_app.exporter import Exporter
from imdb_app.grouping import ImagePayload, group_images
from imdb_app.normalizer import normalize_record
from imdb_app.pipeline import ExtractionPipeline, get_pipeline
from imdb_app.store import ProductStore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(page_title="IMDB Auto-Fill", layout="wide")

PRODUCT_IMAGE_DIR = Path("hackathon_material/Hackathon Materials/product images")


def get_store() -> ProductStore:
    if "store" not in st.session_state:
        st.session_state.store = ProductStore()
    return st.session_state.store


def get_pipeline_instance(provider: str) -> ExtractionPipeline:
    if st.session_state.get("pipeline_provider") != provider:
        st.session_state.pop("pipeline", None)
        st.session_state.pipeline_provider = provider
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline(provider)
    return st.session_state.pipeline


def get_suggestions() -> List[dict]:
    return st.session_state.setdefault("suggestions", [])


def set_suggestions(suggestions: List[dict]) -> None:
    st.session_state.suggestions = suggestions


def available_sample_groups(image_dir: Path = PRODUCT_IMAGE_DIR) -> list[str]:
    if not image_dir.exists():
        return []
    return sorted({path.name.split("_")[0] for path in image_dir.glob("*.jpg") if "_" in path.name})


def load_sample_group(group_id: str, image_dir: Path = PRODUCT_IMAGE_DIR) -> list[ImagePayload]:
    return [
        ImagePayload(filename=path.name, image_bytes=path.read_bytes())
        for path in sorted(image_dir.glob(f"{group_id}_*.jpg"))
    ]


def load_all_sample_payloads(image_dir: Path = PRODUCT_IMAGE_DIR) -> list[ImagePayload]:
    return [ImagePayload(filename=path.name, image_bytes=path.read_bytes()) for path in sorted(image_dir.glob("*.jpg"))]


async def _process_groups_async(groups: list, pipeline: ExtractionPipeline) -> list[tuple[str, ProductRecord | None, Exception | None]]:
    semaphore = asyncio.Semaphore(max(1, settings.group_processing_concurrency))

    async def run_group(group) -> tuple[str, ProductRecord | None, Exception | None]:
        async with semaphore:
            try:
                record = await pipeline.process_group(group)
            except Exception as exc:  # pragma: no cover - network/provider behavior
                return group.group_id, None, exc
            return group.group_id, record, None

    return await asyncio.gather(*(run_group(group) for group in groups))


def process_image_payloads(payloads: Iterable[ImagePayload], pipeline: ExtractionPipeline, store: ProductStore) -> tuple[list[ProductRecord], list[str]]:
    processed: list[ProductRecord] = []
    errors: list[str] = []
    groups = group_images(payloads)

    status = st.status("Processing product groups...", expanded=True)
    status.write(
        f"Queued {len(groups)} product group(s) with concurrency {max(1, settings.group_processing_concurrency)}"
    )
    results = asyncio.run(_process_groups_async(groups, pipeline))
    for group, (_, record, error) in zip(groups, results):
        if error is not None:  # pragma: no cover - defensive logging for user feedback
            errors.append(f"{group.group_id}: {error}")
            status.write(f"Failed {group.group_id}")
        elif record is not None:
            store.upsert(record)
            processed.append(record)
            status.write(f"Extracted {group.group_id} from {len(group.images)} image(s)")
    status.update(label="Extraction complete", state="complete")

    return processed, errors


def process_uploads(files: Iterable["UploadedFile"], pipeline: ExtractionPipeline, store: ProductStore) -> tuple[list[ProductRecord], list[str]]:
    errors: list[str] = []

    if not files:
        return [], errors

    image_payloads: list[ImagePayload] = []
    for upload in files:
        image_bytes = upload.getvalue()
        if image_bytes:
            image_payloads.append(ImagePayload(filename=upload.name, image_bytes=image_bytes))
        else:
            errors.append(f"{upload.name}: empty file")

    processed, processing_errors = process_image_payloads(image_payloads, pipeline, store)
    errors.extend(processing_errors)

    return processed, errors


def build_summary_frame(records: Iterable[ProductRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {"Record": format_record_name(record), "Group": record.filename or "—", "Images": len(record.filenames)}
        for attr in IMDB_ATTRIBUTES:
            row[f"{attr} (value)"] = getattr(record, attr).value
            row[f"{attr} (confidence)"] = getattr(record, attr).confidence
        rows.append(row)
    return pd.DataFrame(rows)


def format_record_name(record: ProductRecord) -> str:
    return record.item_name.value or record.brand.value or record.filename or record.id[:8]


def render_record_editor(record: ProductRecord, threshold: float) -> None:
    title = format_record_name(record)
    with st.expander(title, expanded=False):
        st.caption(f"Record ID: {record.id}")
        for attr in IMDB_ATTRIBUTES:
            attribute = getattr(record, attr)
            label = attr.replace("_", " ").upper()
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
                normalize_record(record)

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


def render_row_controls(records: List[ProductRecord], store: ProductStore) -> None:
    if not records:
        return

    st.subheader("Split / Merge Rows")
    split_col, merge_col = st.columns(2)

    with split_col:
        split_options = {format_record_name(record): record.id for record in records}
        selected_label = st.selectbox("Duplicate a row for manual split", options=list(split_options), key="split-row")
        if st.button("Create split row", disabled=not selected_label):
            source = next(record for record in records if record.id == split_options[selected_label])
            clone = copy.deepcopy(source)
            clone.id = uuid.uuid4().hex
            clone.filename = f"{source.filename or source.id}-split"
            clone.metadata = {**source.metadata, "split_from": source.id}
            store.upsert(clone)
            st.toast("Created split row")
            st.rerun()

    with merge_col:
        merge_options = {f"{format_record_name(record)} ({record.id[:6]})": record.id for record in records}
        selected = st.multiselect("Merge rows into the first selected row", options=list(merge_options), key="merge-rows")
        if st.button("Merge selected", disabled=len(selected) < 2):
            target_id = merge_options[selected[0]]
            target = next(record for record in records if record.id == target_id)
            for label in selected[1:]:
                source = store.remove(merge_options[label])
                if source:
                    target.merge_with(source)
            normalize_record(target)
            st.toast("Merged rows")
            st.rerun()


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


def render_evaluation(records: List[ProductRecord]) -> None:
    st.subheader("Ground Truth Evaluation")
    if not GROUND_TRUTH_PATH.exists():
        st.caption("Ground-truth workbook not found.")
        return
    if not records:
        st.caption("Run extraction before evaluating predictions.")
        return

    report = evaluate_records(records)
    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{report.row_count}/{report.expected_row_count}")
    metrics[1].metric("Exact", f"{report.exact_accuracy:.0%}")
    metrics[2].metric("Normalized", f"{report.normalized_accuracy:.0%}")
    metrics[3].metric("Columns", str(len(EXPORT_COLUMNS)))

    frame = report.to_frame()
    st.dataframe(
        frame.style.format({"Exact": "{:.0%}", "Normalized": "{:.0%}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("Hackathon IMDB Auto-Fill")
    st.caption("Group product images, auto-fill the required 13 columns, review low-confidence fields, and export predictions.")

    store = get_store()
    exporter = st.session_state.setdefault("exporter", Exporter())

    st.sidebar.subheader("Configuration")
    configured_provider = settings.vlm_provider.strip().lower()
    default_provider = configured_provider if configured_provider in {"cohere", "openai"} else "cohere"
    provider = st.sidebar.radio(
        "Model provider",
        options=["cohere", "openai"],
        index=["cohere", "openai"].index(default_provider),
        horizontal=True,
    )
    pipeline = get_pipeline_instance(provider)

    threshold = st.sidebar.slider(
        "Low confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.default_confidence_threshold,
        step=0.05,
    )

    active_key = settings.cohere_api_key if provider == "cohere" else settings.openai_api_key
    active_model = settings.cohere_model if provider == "cohere" else settings.openai_model
    if active_key:
        st.sidebar.success(f"{provider.title()} key detected")
    else:
        st.sidebar.error(f"No {provider.title()} API key provided. Extraction will fail until key is configured.")
    st.sidebar.caption(f"Model: {active_model}")

    if st.sidebar.button("Clear workspace", type="secondary", use_container_width=True):
        store.clear()
        st.session_state.store = ProductStore()
        st.session_state.pop("suggestions", None)
        st.session_state.pop("last_export_path", None)
        st.rerun()

    sample_groups = available_sample_groups()
    if sample_groups:
        st.sidebar.subheader("Sample Images")
        selected_group = st.sidebar.selectbox(
            "Product group",
            options=sample_groups,
            index=sample_groups.index("S227303151") if "S227303151" in sample_groups else 0,
        )
        if st.sidebar.button("Load sample group", use_container_width=True):
            payloads = load_sample_group(selected_group)
            processed, errors = process_image_payloads(payloads, pipeline, store)
            if processed:
                st.success(f"Processed {len(processed)} sample product group(s).")
            if errors:
                error_buffer = io.StringIO()
                for item in errors:
                    error_buffer.write(f"- {item}\n")
                st.error(error_buffer.getvalue())
            set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()]))
        if st.sidebar.button("Load all", use_container_width=True):
            payloads = load_all_sample_payloads()
            processed, errors = process_image_payloads(payloads, pipeline, store)
            if processed:
                st.success(f"Processed {len(processed)} sample product group(s).")
            if errors:
                error_buffer = io.StringIO()
                for item in errors:
                    error_buffer.write(f"- {item}\n")
                st.error(error_buffer.getvalue())
            set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()]))

    uploaded_files = st.file_uploader(
        "Upload product images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="uploader"
    )

    trigger_extraction = st.button("Run grouped extraction", disabled=not uploaded_files)
    if trigger_extraction and uploaded_files:
        processed, errors = process_uploads(uploaded_files, pipeline, store)
        if processed:
            st.success(f"Processed {len(processed)} product group(s).")
        if errors:
            error_buffer = io.StringIO()
            for item in errors:
                error_buffer.write(f"• {item}\n")
            st.error(error_buffer.getvalue())
        set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()]))

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

        render_row_controls(records, store)

        st.subheader("Review & Edit")
        for record in records:
            render_record_editor(record, threshold)

        render_merge_suggestions(records, get_suggestions())
        render_evaluation(records)
    else:
        st.info("Upload product imagery to begin grouped extraction.")


if __name__ == "__main__":
    main()
