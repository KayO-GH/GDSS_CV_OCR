"""Streamlit application for IMDB Auto-Fill."""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import pandas as pd
import streamlit as st

from imdb_app import EXPORT_COLUMNS, IMDB_ATTRIBUTES, ProductRecord, settings
from imdb_app.costing import format_cost_usd, summarize_model_usage
from imdb_app.evaluator import GROUND_TRUTH_PATH, evaluate_aligned_records, evaluate_records
from imdb_app.exporter import Exporter
from imdb_app.grouping import (
    evidence_payload_id,
    ImageEvidence,
    ImageGroup,
    ImagePayload,
    ProductImageCluster,
    group_images_by_filename_prefix,
    infer_product_groups,
)
from imdb_app.model_catalog import (
    ModelProfile,
    available_model_profiles,
    get_model_profile,
    resolve_default_model_key,
    selected_or_first_available,
)
from imdb_app.normalizer import normalize_record
from imdb_app.pack_parser import parse_pack_text
from imdb_app.pipeline import ExtractionPipeline, get_pipeline
from imdb_app.store import ProductStore
from imdb_app.validators import validate_barcode

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(page_title="IMDB Auto-Fill", layout="wide")

REQUIRED_ATTRIBUTES = [
    "item_name",
    "barcode",
    "manufacturer",
    "brand",
    "weight",
    "packaging_type",
    "country",
    "type",
]
MERCHANDISING_ATTRIBUTES = ["variant", "fragrance_flavor", "promotion", "addons", "tagline"]


@dataclass
class BatchRun:
    id: str
    uploaded_filenames: list[str]
    image_count: int
    inferred_groups: list[str] = field(default_factory=list)
    reviewed_groups: list[str] = field(default_factory=list)
    processed_group_ids: list[str] = field(default_factory=list)
    failed_group_ids: list[str] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "identified"
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))

    @property
    def group_count(self) -> int:
        return len(self.reviewed_groups or self.inferred_groups)


def get_store() -> ProductStore:
    if "store" not in st.session_state:
        st.session_state.store = ProductStore()
    return st.session_state.store


def get_pipeline_instance(model_key: str) -> ExtractionPipeline:
    if st.session_state.get("pipeline_model_key") != model_key:
        st.session_state.pop("pipeline", None)
        st.session_state.pipeline_model_key = model_key
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline(model_key)
    return st.session_state.pipeline


def get_suggestions() -> list[dict]:
    return st.session_state.setdefault("suggestions", [])


def set_suggestions(suggestions: list[dict]) -> None:
    st.session_state.suggestions = suggestions


def image_payload_cache() -> dict[str, list[ImagePayload]]:
    return st.session_state.setdefault("image_payloads_by_group", {})


def grouping_evidence_cache() -> dict[str, ImageEvidence]:
    return st.session_state.setdefault("grouping_evidence_by_hash", {})


def upload_payload_cache() -> list[ImagePayload]:
    return st.session_state.setdefault("uploaded_image_payloads", [])


def inferred_cluster_cache() -> list[ProductImageCluster]:
    return st.session_state.setdefault("inferred_image_clusters", [])


def batch_history() -> list[BatchRun]:
    return st.session_state.setdefault("batch_runs", [])


def failed_cluster_cache() -> dict[str, list[ProductImageCluster]]:
    return st.session_state.setdefault("batch_failed_clusters_by_id", {})


def current_batch_id() -> str | None:
    return st.session_state.get("current_batch_id")


def _touch_batch(batch: BatchRun) -> None:
    batch.updated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _next_batch_id() -> str:
    return f"batch-{len(batch_history()) + 1:03d}"


def find_batch(batch_id: str | None) -> BatchRun | None:
    if not batch_id:
        return None
    return next((batch for batch in batch_history() if batch.id == batch_id), None)


def create_batch_run(payloads: list[ImagePayload], clusters: list[ProductImageCluster]) -> BatchRun:
    batch = BatchRun(
        id=_next_batch_id(),
        uploaded_filenames=[payload.filename for payload in payloads],
        image_count=len(payloads),
        inferred_groups=[cluster.group_id for cluster in clusters],
    )
    batch_history().append(batch)
    st.session_state.current_batch_id = batch.id
    failed_cluster_cache().pop(batch.id, None)
    return batch


def update_batch_review(batch_id: str | None, clusters: list[ProductImageCluster]) -> None:
    batch = find_batch(batch_id)
    if batch is None:
        return
    batch.reviewed_groups = [cluster.group_id for cluster in clusters]
    batch.status = "reviewed"
    _touch_batch(batch)


def record_batch_results(
    batch_id: str | None,
    clusters: list[ProductImageCluster],
    processed: list[ProductRecord],
    errors: list[str],
) -> None:
    batch = find_batch(batch_id)
    if batch is None:
        return

    processed_group_ids = {record.filename or record.id for record in processed}
    failed_group_ids = {error.split(":", 1)[0] for error in errors}
    batch.reviewed_groups = [cluster.group_id for cluster in clusters]
    batch.processed_group_ids = sorted(set(batch.processed_group_ids) | processed_group_ids)
    batch.failed_group_ids = sorted(failed_group_ids)
    batch.record_ids = sorted(set(batch.record_ids) | {record.id for record in processed})
    batch.errors = errors
    if failed_group_ids and processed_group_ids:
        batch.status = "partial"
    elif failed_group_ids:
        batch.status = "failed"
    else:
        batch.status = "processed"
    failed_cluster_cache()[batch.id] = [cluster for cluster in clusters if cluster.group_id in failed_group_ids]
    if not failed_group_ids:
        failed_cluster_cache().pop(batch.id, None)
    _touch_batch(batch)


def failed_clusters_for_batch(batch_id: str | None) -> list[ProductImageCluster]:
    if not batch_id:
        return []
    return failed_cluster_cache().get(batch_id, [])


def remember_payloads(payloads: Iterable[ImagePayload]) -> list:
    groups = group_images_by_filename_prefix(payloads)
    cache = image_payload_cache()
    for group in groups:
        cache[group.group_id] = group.images
    return groups


def remember_clusters(clusters: Iterable[ProductImageCluster]) -> list[ImageGroup]:
    groups = [cluster.to_image_group() for cluster in clusters]
    cache = image_payload_cache()
    for group in groups:
        cache[group.group_id] = group.images
    return groups


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
    groups = remember_payloads(payloads)

    status = st.status("Extracting product groups", expanded=True)
    status.write(f"Preprocessing images for {len(groups)} product group(s)")
    status.write("Reading labels and barcode evidence")
    results = asyncio.run(_process_groups_async(groups, pipeline))
    status.write("Normalizing fields and checking validation rules")
    for group, (_, record, error) in zip(groups, results):
        if error is not None:  # pragma: no cover - defensive logging for user feedback
            errors.append(f"{group.group_id}: {error}")
            status.write(f"Failed {group.group_id}")
        elif record is not None:
            store.upsert(record)
            processed.append(record)
            status.write(f"Extracted {group.group_id} from {len(group.images)} image(s)")
    status.write("Checking duplicates")
    status.update(label="Ready for review", state="complete")

    return processed, errors


def process_reviewed_clusters(
    clusters: Iterable[ProductImageCluster],
    pipeline: ExtractionPipeline,
    store: ProductStore,
    *,
    batch_id: str | None = None,
) -> tuple[list[ProductRecord], list[str]]:
    processed: list[ProductRecord] = []
    errors: list[str] = []
    cluster_list = list(clusters)
    update_batch_review(batch_id, cluster_list)
    groups = remember_clusters(cluster_list)

    status = st.status("Extracting reviewed product groups", expanded=True)
    status.write(f"Queued {len(groups)} reviewed product group(s)")
    results = asyncio.run(_process_groups_async(groups, pipeline))
    status.write("Normalizing fields and checking validation rules")
    for group, (_, record, error) in zip(groups, results):
        if error is not None:  # pragma: no cover - defensive logging for user feedback
            errors.append(f"{group.group_id}: {error}")
            status.write(f"Failed {group.group_id}")
        elif record is not None:
            if batch_id:
                record.metadata["batch_id"] = batch_id
            store.upsert(record)
            processed.append(record)
            status.write(f"Extracted {group.group_id} from {len(group.images)} image(s)")
    status.write("Checking duplicates")
    status.update(label="Ready for field review", state="complete")
    record_batch_results(batch_id, cluster_list, processed, errors)

    return processed, errors


def payloads_from_uploads(files: Iterable["UploadedFile"]) -> tuple[list[ImagePayload], list[str]]:
    payloads: list[ImagePayload] = []
    errors: list[str] = []
    for upload in files:
        image_bytes = upload.getvalue()
        if image_bytes:
            payloads.append(ImagePayload(filename=upload.name, image_bytes=image_bytes))
        else:
            errors.append(f"{upload.name}: empty file")
    return payloads, errors


def prefix_group_clusters(payloads: Iterable[ImagePayload]) -> list[ProductImageCluster]:
    groups = group_images_by_filename_prefix(payloads)
    if not any(len(group.images) > 1 for group in groups):
        return []

    return [
        ProductImageCluster(
            group_id=group.group_id,
            images=group.images,
            evidence=[],
            confidence=1.0,
            reason="matching filename prefix",
            needs_review=False,
        )
        for group in groups
    ]


def identify_product_groups(
    payloads: list[ImagePayload],
    pipeline: ExtractionPipeline,
    *,
    consider_filename_prefixes: bool,
) -> tuple[list[ProductImageCluster], list[str]]:
    errors: list[str] = []
    if not payloads:
        return [], errors

    st.session_state.uploaded_image_payloads = payloads
    status = st.status("Identifying product groups", expanded=True)
    if len(payloads) == 1:
        clusters = [
            ProductImageCluster(
                group_id="review-001",
                images=[payloads[0]],
                evidence=[],
                confidence=0.5,
                reason="single uploaded image; no grouping needed",
                needs_review=True,
            )
        ]
        st.session_state.inferred_image_clusters = clusters
        create_batch_run(payloads, clusters)
        status.write("Single image queued for review")
        status.update(label="Single image queued for review", state="complete")
        return clusters, errors

    if consider_filename_prefixes:
        clusters = prefix_group_clusters(payloads)
        if clusters:
            st.session_state.inferred_image_clusters = clusters
            create_batch_run(payloads, clusters)
            status.write("Grouped by filename prefix")
            status.write(f"Created {len(clusters)} candidate product group(s)")
            status.update(label="Grouped by filename prefix", state="complete")
            return clusters, errors

    status.write(f"Analyzing {len(payloads)} image(s) independently")
    try:
        evidence = asyncio.run(pipeline.analyze_images_for_grouping(payloads, grouping_evidence_cache()))
    except Exception as exc:  # pragma: no cover - provider/network behavior
        status.update(label="Grouping failed", state="error")
        return [], [str(exc)]

    evidence_by_payload_id = {item.payload_id: item for item in evidence}
    clusters = infer_product_groups(payloads, evidence_by_payload_id)
    st.session_state.inferred_image_clusters = clusters
    create_batch_run(payloads, clusters)
    status.write("Grouped by image evidence")
    status.write(f"Created {len(clusters)} candidate product group(s)")
    status.update(label="Grouped by image evidence", state="complete")
    return clusters, errors


def build_summary_frame(records: Iterable[ProductRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {"Record": format_record_name(record), "Group": record.filename or "-", "Images": len(record.filenames)}
        for attr in IMDB_ATTRIBUTES:
            row[f"{attr} (value)"] = getattr(record, attr).value
            row[f"{attr} (confidence)"] = getattr(record, attr).confidence
        rows.append(row)
    return pd.DataFrame(rows)


def export_frame(records: Iterable[ProductRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.values_for_export() for record in records], columns=EXPORT_COLUMNS).fillna("")


def format_record_name(record: ProductRecord) -> str:
    return record.item_name.value or record.brand.value or record.filename or record.id[:8]


def field_label(attr: str) -> str:
    return attr.replace("_", " ").upper()


def render_shell_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
        .gdss-step {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            background: rgba(250, 250, 250, 0.72);
        }
        .gdss-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.65rem;
            background: white;
        }
        .gdss-muted {color: rgba(49, 51, 63, 0.7); font-size: 0.9rem;}
        .gdss-issue {color: #8a4b00; font-weight: 600;}
        @media (max-width: 720px) {
            .block-container {padding-left: 0.75rem; padding-right: 0.75rem;}
            .gdss-step {padding: 0.75rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def recompute_suggestions(store: ProductStore) -> None:
    set_suggestions(store.merge_suggestions([record.to_dict() for record in store.all()]))


def render_header(records: list[ProductRecord], model_profile: ModelProfile, active_key: str | None) -> None:
    st.title("IMDB Auto-Fill")
    st.caption("Product photos to reviewed, validated, database-ready item-master rows.")
    cols = st.columns(4)
    cols[0].metric("Rows", len(records))
    cols[1].metric("Required complete", f"{required_completion(records):.0%}" if records else "0%")
    cols[2].metric("Open issues", count_review_issues(records))
    cols[3].metric("Model", model_profile.provider_label, "key detected" if active_key else "missing key")
    cols[3].caption(model_profile.model_id)


def required_completion(records: list[ProductRecord]) -> float:
    if not records:
        return 0.0
    total = len(records) * len(REQUIRED_ATTRIBUTES)
    complete = sum(1 for record in records for attr in REQUIRED_ATTRIBUTES if getattr(record, attr).value)
    return complete / total if total else 0.0


def count_review_issues(records: list[ProductRecord], threshold: float | None = None) -> int:
    threshold = settings.default_confidence_threshold if threshold is None else threshold
    issues = 0
    for record in records:
        for attr in REQUIRED_ATTRIBUTES:
            attribute = getattr(record, attr)
            if not attribute.value or (attribute.confidence is not None and attribute.confidence < threshold):
                issues += 1
        barcode_validation = validate_barcode(record.barcode.value)
        if record.barcode.value and not barcode_validation.is_valid:
            issues += 1
        if record.metadata.get("barcode_conflict"):
            issues += 1
    return issues


def render_image_strip(record: ProductRecord) -> None:
    payloads = image_payload_cache().get(record.filename or "", [])
    if not payloads:
        st.caption("No product image preview is available for this row.")
        return

    for payload in payloads[:4]:
        st.image(payload.image_bytes, caption=payload.filename, width=170)
    if len(payloads) > 4:
        st.caption(f"{len(payloads) - 4} more image(s) in this group")


def render_field_editor(record: ProductRecord, attr: str, threshold: float) -> None:
    attribute = getattr(record, attr)
    confidence = attribute.confidence or 0.0
    has_barcode_conflict = attr == "barcode" and bool(record.metadata.get("barcode_conflict"))
    needs_review = bool(not attribute.value or confidence < threshold or has_barcode_conflict)
    title = f"{field_label(attr)}"
    if needs_review:
        title += " - needs review"

    with st.expander(title, expanded=needs_review):
        value = attribute.value or ""
        key = f"record-{record.id}-{attr}"
        new_value = st.text_input(field_label(attr), value=value, key=key)
        cleaned_value = new_value.strip() or None
        if cleaned_value != attribute.value:
            attribute.value = cleaned_value
            if cleaned_value:
                attribute.source = "manual_edit"
                attribute.confidence = 1.0
            record.metadata.setdefault("edited", {})[attr] = True
            normalize_record(record)

        details = [f"Confidence {confidence:.0%}"]
        if attribute.source:
            details.append(f"Source: {attribute.source}")
        if attribute.notes:
            details.append(f"Notes: {attribute.notes}")
        if attr == "barcode" and record.metadata.get("barcode_conflict"):
            details.append("Conflict: scanner and model disagreed")

        message = " | ".join(details)
        if needs_review:
            st.markdown(f"<span class='gdss-issue'>{message}</span>", unsafe_allow_html=True)
        else:
            st.caption(message)

        if attr == "barcode":
            candidates = record.metadata.get("barcode_candidates") or []
            if candidates:
                st.markdown("Barcode candidates")
                frame = pd.DataFrame(candidates)
                visible_columns = [
                    column
                    for column in ["value", "type", "is_valid", "reason", "source_image", "quality", "selected"]
                    if column in frame.columns
                ]
                st.dataframe(frame[visible_columns], width="stretch", hide_index=True)


def render_record_workspace(record: ProductRecord, threshold: float) -> None:
    st.markdown(f"#### {format_record_name(record)}")

    image_col, fields_col = st.columns([0.32, 0.68], gap="large")
    with image_col:
        st.markdown("**Images**")
        render_image_strip(record)
        st.caption(f"Group: {record.filename or '-'} | Record ID: {record.id}")

    with fields_col:
        tabs = st.tabs(["Required fields", "Merchandising", "Metadata"])
        with tabs[0]:
            for attr in REQUIRED_ATTRIBUTES:
                render_field_editor(record, attr, threshold)
        with tabs[1]:
            for attr in MERCHANDISING_ATTRIBUTES:
                render_field_editor(record, attr, threshold)
        with tabs[2]:
            metadata_rows = [{"Key": key, "Value": str(value)} for key, value in sorted(record.metadata.items())]
            st.dataframe(pd.DataFrame(metadata_rows), width="stretch", hide_index=True)


def validation_rows(records: list[ProductRecord]) -> list[dict]:
    rows = []
    for record in records:
        barcode = validate_barcode(record.barcode.value)
        pack = parse_pack_text(record.item_name.value, record.weight.value, record.promotion.value, record.addons.value)
        rows.append(
            {
                "Record": format_record_name(record),
                "Barcode": barcode.reason if record.barcode.value else "Missing barcode",
                "Weight parse": pack.normalized_weight or "No parse",
                "Pack count": pack.pack_count or "",
                "Promotion": record.promotion.value or pack.promotion or "",
                "Add-ons": record.addons.value or pack.addons or "",
                "Barcode candidates": len(record.metadata.get("barcode_candidates") or []),
                "Selected barcode image": (record.metadata.get("selected_barcode_candidate") or {}).get("source_image", ""),
                "Required": f"{sum(1 for attr in REQUIRED_ATTRIBUTES if getattr(record, attr).value)}/{len(REQUIRED_ATTRIBUTES)}",
                "Conflict": "Yes" if record.metadata.get("barcode_conflict") else "No",
            }
        )
    return rows


def render_merge_suggestions(records: list[ProductRecord], suggestions: list[dict], store: ProductStore) -> None:
    st.markdown("##### Duplicate status")
    if not records:
        st.caption("Duplicate checks will appear after extraction.")
        return
    if not suggestions:
        st.success("No duplicate found")
        return

    id_lookup = {record.id: record for record in records}
    for suggestion in suggestions:
        target = id_lookup.get(suggestion.get("record_id"))
        target_name = format_record_name(target) if target else suggestion.get("record_id")
        with st.container(border=True):
            st.markdown(f"**{target_name}**")
            for candidate in suggestion.get("candidates", []):
                candidate_record = id_lookup.get(candidate.get("candidate_id"))
                candidate_name = format_record_name(candidate_record) if candidate_record else candidate.get("candidate_id")
                reasons = ", ".join(candidate.get("reasons", [])) or "No specific reasons"
                st.write(f"{candidate_name}: score {candidate.get('score', 0):.2f} from {reasons}")
                cols = st.columns(3)
                action_key = f"dup-action-{suggestion.get('record_id')}-{candidate.get('candidate_id')}"
                if cols[0].button("Keep separate", key=f"{action_key}-keep"):
                    st.session_state.setdefault("duplicate_actions", {})[action_key] = "keep_separate"
                    st.toast("Marked as separate")
                if cols[1].button("Mark duplicate", key=f"{action_key}-mark"):
                    st.session_state.setdefault("duplicate_actions", {})[action_key] = "marked_duplicate"
                    if target:
                        target.metadata.setdefault("duplicate_review", {})[candidate.get("candidate_id")] = "marked_duplicate"
                    st.toast("Marked as duplicate")
                if cols[2].button("Merge", key=f"{action_key}-merge", disabled=not target or not candidate_record):
                    if target and candidate_record:
                        target.merge_with(candidate_record)
                        store.remove(candidate_record.id)
                        normalize_record(target)
                        recompute_suggestions(store)
                        st.toast("Merged rows")
                        st.rerun()


def cluster_label(cluster: ProductImageCluster) -> str:
    review = " - needs review" if cluster.needs_review else ""
    return f"{cluster.group_id} ({len(cluster.images)} image(s), {cluster.confidence:.0%}){review}"


def evidence_frame(cluster: ProductImageCluster) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Filename": evidence.filename,
                "Barcode": evidence.barcode or "",
                "Barcode valid": "Yes" if evidence.barcode_is_valid else "No",
                "Brand": evidence.brand or "",
                "Item name": evidence.item_name or "",
                "Weight": evidence.weight or "",
                "Packaging": evidence.packaging_type or "",
                "Type": evidence.type or "",
                "Confidence": evidence.confidence,
            }
            for evidence in cluster.evidence
        ]
    )


def _evidence_for_image(cluster: ProductImageCluster, image: ImagePayload) -> ImageEvidence | None:
    for evidence in cluster.evidence:
        if evidence_payload_id(evidence) == image.payload_id:
            return evidence
    return None


def _make_manual_cluster(group_id: str, images: list[ImagePayload], evidence: list[ImageEvidence], reason: str) -> ProductImageCluster:
    confidence = min([item.confidence for item in evidence], default=0.5)
    return ProductImageCluster(
        group_id=group_id,
        images=sorted(images, key=lambda item: item.filename),
        evidence=sorted(evidence, key=lambda item: item.filename),
        confidence=confidence,
        reason=reason,
        needs_review=False,
    )


def _next_manual_group_id(clusters: Iterable[ProductImageCluster], prefix: str) -> str:
    used_ids = {cluster.group_id for cluster in clusters}
    counter = 1
    while True:
        candidate = f"{prefix}-{counter:03d}"
        if candidate not in used_ids:
            return candidate
        counter += 1


def render_group_review(clusters: list[ProductImageCluster]) -> list[ProductImageCluster]:
    st.markdown("##### Review inferred product groups")
    if not clusters:
        st.caption("Upload images, then identify product groups before extraction.")
        return clusters

    for cluster in clusters:
        with st.container(border=True):
            st.markdown(f"**{cluster_label(cluster)}**")
            st.caption(cluster.reason or "Product evidence grouping")
            preview_cols = st.columns(min(len(cluster.images), 4) or 1)
            for index, image in enumerate(cluster.images[:4]):
                preview_cols[index].image(image.image_bytes, caption=image.filename, width=140)
            if len(cluster.images) > 4:
                st.caption(f"{len(cluster.images) - 4} more image(s) in this candidate group")
            st.dataframe(evidence_frame(cluster), width="stretch", hide_index=True)

    with st.expander("Adjust inferred groups", expanded=any(cluster.needs_review for cluster in clusters)):
        group_options = {cluster_label(cluster): cluster.group_id for cluster in clusters}
        split_col, merge_col, move_col = st.columns(3)

        with split_col:
            split_label = st.selectbox("Split group into single images", options=list(group_options), key="split-upload-cluster")
            if st.button("Split group", disabled=not split_label, width="stretch"):
                split_id = group_options[split_label]
                updated: list[ProductImageCluster] = []
                for cluster in clusters:
                    if cluster.group_id != split_id:
                        updated.append(cluster)
                        continue
                    for image in cluster.images:
                        evidence = _evidence_for_image(cluster, image)
                        updated.append(
                            _make_manual_cluster(
                                _next_manual_group_id([*updated, *clusters], "review-split"),
                                [image],
                                [evidence] if evidence else [],
                                "manually split for review",
                            )
                        )
                st.session_state.inferred_image_clusters = updated
                st.rerun()

        with merge_col:
            merge_labels = st.multiselect("Merge groups", options=list(group_options), key="merge-upload-clusters")
            if st.button("Merge selected groups", disabled=len(merge_labels) < 2, width="stretch"):
                merge_ids = {group_options[label] for label in merge_labels}
                merged_images: list[ImagePayload] = []
                merged_evidence: list[ImageEvidence] = []
                updated = []
                for cluster in clusters:
                    if cluster.group_id in merge_ids:
                        merged_images.extend(cluster.images)
                        merged_evidence.extend(cluster.evidence)
                    else:
                        updated.append(cluster)
                updated.append(
                    _make_manual_cluster(
                        _next_manual_group_id([*updated, *clusters], "auto-merged"),
                        merged_images,
                        merged_evidence,
                        "manually merged by reviewer",
                    )
                )
                st.session_state.inferred_image_clusters = updated
                st.rerun()

        with move_col:
            image_options = {
                f"{image.filename} ({cluster.group_id})": (cluster.group_id, image.payload_id)
                for cluster in clusters
                for image in cluster.images
            }
            move_label = st.selectbox("Move image", options=list(image_options), key="move-upload-image")
            target_label = st.selectbox("Target group", options=list(group_options), key="move-upload-target")
            if st.button("Move image", disabled=not move_label or not target_label, width="stretch"):
                source_id, payload_id = image_options[move_label]
                target_id = group_options[target_label]
                if source_id != target_id:
                    moved_image: ImagePayload | None = None
                    moved_evidence: ImageEvidence | None = None
                    updated = []
                    for cluster in clusters:
                        if cluster.group_id == source_id:
                            moved_image = next((image for image in cluster.images if image.payload_id == payload_id), None)
                            moved_evidence = next((item for item in cluster.evidence if evidence_payload_id(item) == payload_id), None)
                            break

                    if moved_image is None:
                        st.warning("Selected image could not be moved.")
                        return inferred_cluster_cache()

                    for cluster in clusters:
                        if cluster.group_id == source_id:
                            remaining_images = [image for image in cluster.images if image.payload_id != payload_id]
                            remaining_evidence = [item for item in cluster.evidence if evidence_payload_id(item) != payload_id]
                            if remaining_images:
                                updated.append(
                                    _make_manual_cluster(cluster.group_id, remaining_images, remaining_evidence, cluster.reason)
                                )
                        elif cluster.group_id == target_id:
                            updated.append(
                                _make_manual_cluster(
                                    cluster.group_id,
                                    [*cluster.images, moved_image],
                                    [*cluster.evidence, *([moved_evidence] if moved_evidence else [])],
                                    "manually adjusted by reviewer",
                                )
                            )
                        else:
                            updated.append(cluster)
                    if not any(cluster.group_id == target_id for cluster in updated):
                        updated.append(_make_manual_cluster(target_id, [moved_image], [moved_evidence] if moved_evidence else [], "manual move"))
                    st.session_state.inferred_image_clusters = updated
                    st.rerun()

    return inferred_cluster_cache()


def render_row_controls(records: list[ProductRecord], store: ProductStore) -> None:
    if not records:
        return

    with st.expander("Advanced split / merge tools", expanded=False):
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
                recompute_suggestions(store)
                st.toast("Merged rows")
                st.rerun()


def render_scorecard(records: list[ProductRecord], enable_hackathon_benchmark: bool) -> None:
    if not records:
        st.caption("Run extraction before viewing validation.")
        return

    validation = pd.DataFrame(validation_rows(records))
    valid_barcodes = sum(1 for record in records if validate_barcode(record.barcode.value).is_valid)
    metrics = st.columns(4)
    metrics[0].metric("Field completion", f"{required_completion(records):.0%}")
    metrics[1].metric("Valid barcodes", f"{valid_barcodes}/{len(records)}")
    metrics[2].metric("Needs review", count_review_issues(records))
    metrics[3].metric("Columns", len(EXPORT_COLUMNS))
    st.dataframe(validation, width="stretch", hide_index=True)
    render_model_cost_summary(records)

    if enable_hackathon_benchmark and GROUND_TRUTH_PATH.exists():
        st.markdown("##### Hackathon workbook comparison")
        aligned = evaluate_aligned_records(records)
        row_order = evaluate_records(records)
        eval_cols = st.columns(3)
        eval_cols[0].metric("Ground-truth aligned rows", f"{aligned.aligned_count}/{aligned.row_count}")
        if aligned.aligned_count:
            eval_cols[1].metric("Aligned normalized match", f"{aligned.normalized_accuracy:.0%}")
        else:
            eval_cols[1].metric("Aligned normalized match", "n/a")
        eval_cols[2].metric("Workbook row-order benchmark", f"{row_order.normalized_accuracy:.0%}")
        st.caption("Ground-truth match is shown only for aligned rows. Row-order benchmark is retained for workbook-order comparisons.")


def render_model_cost_summary(records: list[ProductRecord]) -> None:
    summary = summarize_model_usage(records)
    if summary["request_count"] == 0:
        st.caption("Model usage and cost will appear after extraction.")
        return

    st.markdown("##### Model usage and cost")
    cost_cols = st.columns(4)
    cost_cols[0].metric("Model requests", summary["request_count"])
    cost_cols[1].metric("Images analyzed", summary["image_count"])
    cost_cols[2].metric("Tokens", summary["total_tokens"])
    cost_cols[3].metric("Cost", format_cost_usd(summary["total_cost_usd"]) if summary["known_cost_count"] else "n/a")

    models = ", ".join(summary["models"]) or "Unknown model"
    if summary["cost_available"]:
        st.caption(f"Models: {models}. Cost is based on provider usage and configured pricing.")
    elif summary["partial_cost_available"]:
        st.caption(f"Models: {models}. Some records have pricing unavailable.")
    else:
        notes = " ".join(summary["pricing_notes"]) or "Pricing unavailable for the selected model."
        st.caption(f"Models: {models}. {notes}")


def batch_history_frame() -> pd.DataFrame:
    rows = []
    for batch in batch_history():
        rows.append(
            {
                "Batch": batch.id,
                "Status": batch.status,
                "Images": batch.image_count,
                "Groups": batch.group_count,
                "Processed groups": len(batch.processed_group_ids),
                "Failed groups": len(batch.failed_group_ids),
                "Records": len(batch.record_ids),
                "Updated": batch.updated_at,
            }
        )
    return pd.DataFrame(rows)


def render_batch_history() -> None:
    history = batch_history()
    if not history:
        st.caption("Batch runs will appear after product-group identification.")
        return

    st.markdown("##### Batch runs")
    st.dataframe(batch_history_frame(), width="stretch", hide_index=True)
    active = find_batch(current_batch_id())
    if active is not None and active.errors:
        st.warning("\n".join(f"- {item}" for item in active.errors))


def render_export_controls(records: list[ProductRecord], exporter: Exporter) -> None:
    if not records:
        st.caption("Validated rows will appear here after extraction.")
        return

    frame = export_frame(records)
    st.markdown("##### Search-ready output")
    filters = st.columns(4)
    brand = filters[0].selectbox("Brand", ["All"] + sorted(value for value in frame["BRAND"].unique() if value))
    item_type = filters[1].selectbox("Type", ["All"] + sorted(value for value in frame["TYPE"].unique() if value))
    weight = filters[2].selectbox("Weight", ["All"] + sorted(value for value in frame["WEIGHT"].unique() if value))
    packaging = filters[3].selectbox("Packaging", ["All"] + sorted(value for value in frame["PACKAGING  TYPE"].unique() if value))

    filtered = frame.copy()
    for column, value in [("BRAND", brand), ("TYPE", item_type), ("WEIGHT", weight), ("PACKAGING  TYPE", packaging)]:
        if value != "All":
            filtered = filtered[filtered[column] == value]

    st.dataframe(filtered, width="stretch", hide_index=True)

    export_cols = st.columns(2)
    for format_label, column in [("csv", export_cols[0]), ("excel", export_cols[1])]:
        if column.button(f"Generate {format_label.upper()} export", disabled=not records, width="stretch"):
            path = exporter.export(records, format=format_label)
            st.session_state.last_export_path = str(path)
            st.toast(f"Exported {path.name}")

    export_path_str = st.session_state.get("last_export_path")
    if export_path_str:
        export_path = Path(export_path_str)
        if export_path.exists():
            mime = "text/csv" if export_path.suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            st.success(f"Export ready: {export_path.name}")
            st.download_button(
                label=f"Download {export_path.name}",
                data=export_path.read_bytes(),
                file_name=export_path.name,
                mime=mime,
                width="stretch",
            )
            st.caption(str(export_path))


def render_add_images_step(store: ProductStore, pipeline: ExtractionPipeline, consider_filename_prefixes: bool) -> None:
    st.markdown("### 1. Add Images")
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Upload product images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="uploader"
        )
        if uploaded_files:
            preview_payloads, upload_errors = payloads_from_uploads(uploaded_files)
            if upload_errors:
                st.error("\n".join(f"- {item}" for item in upload_errors))
            if consider_filename_prefixes:
                st.caption("Matching file name prefixes are used first, then the app falls back to image evidence grouping.")
            else:
                st.caption("File name prefixes are ignored. Uploaded images are grouped by product evidence.")
            st.write(f"{len(preview_payloads)} image(s) ready for product-group identification.")
            if st.button("Identify product groups", width="stretch"):
                clusters, errors = identify_product_groups(
                    preview_payloads,
                    pipeline,
                    consider_filename_prefixes=consider_filename_prefixes,
                )
                if clusters:
                    st.success(f"Identified {len(clusters)} candidate product group(s). Review them before extraction.")
                if errors:
                    st.error("\n".join(f"- {item}" for item in errors))

        clusters = inferred_cluster_cache()
        if clusters:
            active_batch = find_batch(current_batch_id())
            if active_batch is not None:
                st.caption(f"Current batch: {active_batch.id} | {active_batch.status}")
            reviewed_clusters = render_group_review(clusters)
            if st.button("Run extraction for reviewed groups", type="primary", width="stretch"):
                processed, errors = process_reviewed_clusters(reviewed_clusters, pipeline, store, batch_id=current_batch_id())
                if processed:
                    st.success(f"Processed {len(processed)} reviewed product group(s).")
                if errors:
                    st.error("\n".join(f"- {item}" for item in errors))
                recompute_suggestions(store)
                st.rerun()
            retry_clusters = failed_clusters_for_batch(current_batch_id())
            if retry_clusters and st.button("Retry failed groups", width="stretch"):
                processed, errors = process_reviewed_clusters(retry_clusters, pipeline, store, batch_id=current_batch_id())
                if processed:
                    st.success(f"Retried {len(processed)} failed product group(s).")
                if errors:
                    st.error("\n".join(f"- {item}" for item in errors))
                recompute_suggestions(store)
                st.rerun()


def render_workflow(
    records: list[ProductRecord],
    threshold: float,
    store: ProductStore,
    exporter: Exporter,
    enable_hackathon_benchmark: bool,
) -> None:
    st.markdown("### 2. Extract")
    with st.container(border=True):
        if records:
            st.success(f"{len(records)} row(s) ready for review.")
        else:
            st.info("Upload product photos to start extraction.")
        render_batch_history()

    st.markdown("### 3. Review Fields")
    with st.container(border=True):
        if records:
            for record in records:
                render_record_workspace(record, threshold)
                st.divider()
            render_row_controls(records, store)
        else:
            st.caption("Field cards will appear here after Step 1.")

    st.markdown("### 4. Validate & Deduplicate")
    with st.container(border=True):
        render_scorecard(records, enable_hackathon_benchmark)
        render_merge_suggestions(records, get_suggestions(), store)

    st.markdown("### 5. Export")
    with st.container(border=True):
        render_export_controls(records, exporter)


def render_sidebar(model_key: str) -> tuple[ExtractionPipeline, float, bool, bool, str | None, str]:
    st.sidebar.subheader("Advanced configuration")
    profile = get_model_profile(model_key)
    pipeline = get_pipeline_instance(profile.key)
    threshold = st.sidebar.slider(
        "Low confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.default_confidence_threshold,
        step=0.05,
    )
    consider_filename_prefixes = st.sidebar.toggle("Consider file name prefixes", value=True)
    enable_hackathon_benchmark = st.sidebar.toggle("Enable hackathon benchmark", value=False)
    active_key = profile.credential_value
    if active_key:
        st.sidebar.success(f"{profile.provider_label} key detected")
    else:
        st.sidebar.warning(f"No {profile.provider_label} key. Configure a key before running live extraction.")
    st.sidebar.caption(f"Model: {profile.model_id}")
    return pipeline, threshold, consider_filename_prefixes, enable_hackathon_benchmark, active_key, profile.label


def visible_model_profiles() -> list:
    profiles = available_model_profiles(include_unavailable_visible=False)
    if profiles:
        return profiles
    return [selected_or_first_available(resolve_default_model_key())]


def default_visible_model_profile(profiles: list):
    configured_default = (settings.default_model_key or "").strip()
    if configured_default:
        selected = selected_or_first_available(configured_default)
        if any(profile.key == selected.key for profile in profiles):
            return selected
    return profiles[0]


def main() -> None:
    render_shell_styles()
    store = get_store()
    exporter = st.session_state.setdefault("exporter", Exporter())

    profiles = visible_model_profiles()
    profile_labels = [profile.label for profile in profiles]
    default_profile = default_visible_model_profile(profiles)
    if st.session_state.get("selected_model_label") not in profile_labels:
        st.session_state["selected_model_label"] = default_profile.label
    selected_label = st.sidebar.selectbox(
        "Extraction model",
        options=profile_labels,
        key="selected_model_label",
    )
    selected_profile = next(profile for profile in profiles if profile.label == selected_label)
    pipeline, threshold, consider_filename_prefixes, enable_hackathon_benchmark, active_key, _model_label = render_sidebar(
        selected_profile.key
    )

    if st.sidebar.button("Clear workspace", type="secondary", width="stretch"):
        store.clear()
        st.session_state.store = ProductStore()
        st.session_state.pop("suggestions", None)
        st.session_state.pop("last_export_path", None)
        st.session_state.pop("image_payloads_by_group", None)
        st.session_state.pop("grouping_evidence_by_hash", None)
        st.session_state.pop("uploaded_image_payloads", None)
        st.session_state.pop("inferred_image_clusters", None)
        st.session_state.pop("batch_runs", None)
        st.session_state.pop("current_batch_id", None)
        st.session_state.pop("batch_failed_clusters_by_id", None)
        st.rerun()

    records = store.all()
    if st.sidebar.button("Recompute duplicates", disabled=not records, width="stretch"):
        recompute_suggestions(store)
        st.rerun()

    render_header(records, selected_profile, active_key)
    render_add_images_step(store, pipeline, consider_filename_prefixes)
    render_workflow(store.all(), threshold, store, exporter, enable_hackathon_benchmark)


if __name__ == "__main__":
    main()
