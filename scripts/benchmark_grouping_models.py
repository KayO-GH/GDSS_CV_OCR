"""Benchmark hackathon image grouping quality for local or remote model pipelines."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from typing import Any

import httpx

from imdb_app.barcode import extract_barcode
from imdb_app.config import settings
from imdb_app.grouping import ImageEvidence, ImageGroup, infer_product_groups
from imdb_app.grouping_benchmark import (
    PairwiseGroupingReport,
    evaluate_group_predictions,
    load_hackathon_image_payloads,
    truth_groups_from_filenames,
)
from imdb_app.pipeline import ExtractionPipeline, preprocess
from imdb_app.validators import validate_barcode
from imdb_app.vlm_client import (
    CohereVLMClient,
    HuggingFaceRouterVLMClient,
    OpenAIVLMClient,
    build_attribute_schema,
    get_vlm_client,
    hf_router_response_format,
    image_data_url,
)


def _report_row(name: str, report: PairwiseGroupingReport) -> dict[str, object]:
    return {
        "name": name,
        "images": report.image_count,
        "truth_groups": report.truth_group_count,
        "predicted_groups": report.predicted_group_count,
        "precision": round(report.precision, 4),
        "recall": round(report.recall, 4),
        "f1": round(report.f1, 4),
        "exact_group_matches": report.exact_group_matches,
        "true_positive_pairs": report.true_positive_pairs,
        "false_positive_pairs": report.false_positive_pairs,
        "false_negative_pairs": report.false_negative_pairs,
    }


def _chunked(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _batch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "barcode": {"type": ["string", "null"]},
                        "brand": {"type": ["string", "null"]},
                        "item_name": {"type": ["string", "null"]},
                        "weight": {"type": ["string", "null"]},
                        "packaging_type": {"type": ["string", "null"]},
                        "type": {"type": ["string", "null"]},
                        "confidence": {"type": ["number", "null"]},
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": [
                        "filename",
                        "barcode",
                        "brand",
                        "item_name",
                        "weight",
                        "packaging_type",
                        "type",
                        "confidence",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["images"],
        "additionalProperties": False,
    }


def _batch_prompt() -> str:
    return (
        "Analyze each image independently. Do not group images. "
        "For every image filename, return one object with filename, barcode, brand, item_name, "
        "weight, packaging_type, type, confidence, and notes. "
        "Use null for unknown fields. BARCODE should be digits only when visible."
    )


def _parse_text_json(response: dict[str, Any]) -> dict[str, Any]:
    def decode_json(text: str) -> Any:
        decoder = json.JSONDecoder()
        remainder = text.strip()
        values: list[Any] = []
        while remainder:
            value, end_index = decoder.raw_decode(remainder)
            values.append(value)
            remainder = remainder[end_index:].lstrip()
        if not values:
            msg = "No JSON content found in model response text"
            raise RuntimeError(msg)
        return values[-1]

    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                return decode_json(content)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
                        return decode_json(item["text"])

    message = response.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                return decode_json(item["text"])
    elif isinstance(content, str):
        return decode_json(content)
    if isinstance(response.get("text"), str):
        return decode_json(response["text"])
    if isinstance(response.get("output"), str):
        return decode_json(response["output"])
    msg = "Could not parse JSON payload from model response"
    raise RuntimeError(msg)


def _build_batch_payload(client: Any, batch: list[tuple[str, bytes]], batch_id: str) -> dict[str, Any]:
    schema = _batch_schema()
    if isinstance(client, CohereVLMClient):
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": _batch_prompt()},
            {"type": "text", "text": f"Batch: {batch_id}"},
        ]
        for filename, image_bytes in batch:
            user_content.extend(
                [
                    {"type": "text", "text": f"Image filename: {filename}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_bytes), "detail": "high"}},
                ]
            )
        return {
            "model": client.model,
            "messages": [{"role": "user", "content": user_content}],
            "response_format": {"type": "json_object", "schema": schema},
        }

    if isinstance(client, (HuggingFaceRouterVLMClient, OpenAIVLMClient)):
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Batch: {batch_id}"},
            {"type": "text", "text": _batch_prompt()},
        ]
        for filename, image_bytes in batch:
            user_content.extend(
                [
                    {"type": "text", "text": f"Image filename: {filename}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_bytes)}},
                ]
            )
        response_format: dict[str, Any]
        if isinstance(client, HuggingFaceRouterVLMClient):
            response_format = hf_router_response_format(
                client.model,
                schema_name="grouping_batch_schema",
                schema=schema,
            )
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "grouping_batch_schema", "schema": schema, "strict": True},
            }
        return {
            "model": client.model,
            "messages": [
                {"role": "system", "content": "You are a retail product image analyst. Return strict JSON only."},
                {"role": "user", "content": user_content},
            ],
            "response_format": response_format,
        }

    msg = f"Unsupported client type: {type(client).__name__}"
    raise ValueError(msg)


async def _extract_batch_rows(client: Any, batch: list[tuple[str, bytes]], batch_id: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"}
    payload = _build_batch_payload(client, batch, batch_id)
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as raw_client:
        response = await client._post_with_retries(raw_client, headers=headers, payload=payload, group_id=batch_id)
    parsed = _parse_text_json(response.json())
    if isinstance(parsed, list):
        rows = parsed
    else:
        rows = parsed.get("images") or parsed.get("items") or parsed.get("results")
    if not isinstance(rows, list):
        msg = f"Model returned invalid batch payload for {batch_id}"
        raise RuntimeError(msg)
    return rows


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _evidence_from_row(payload, row: dict[str, Any]) -> ImageEvidence:
    scanner_validation = validate_barcode(extract_barcode(payload.image_bytes))
    model_validation = validate_barcode(row.get("barcode"))
    chosen_validation = scanner_validation if scanner_validation.is_valid else model_validation
    chosen_barcode = chosen_validation.value or _normalize_text(row.get("barcode"))
    confidence = row.get("confidence")
    return ImageEvidence(
        payload_id=payload.payload_id,
        filename=payload.filename,
        image_hash="",
        barcode=chosen_barcode,
        barcode_is_valid=chosen_validation.is_valid,
        barcode_type=chosen_validation.barcode_type,
        item_name=_normalize_text(row.get("item_name")),
        brand=_normalize_text(row.get("brand")),
        weight=_normalize_text(row.get("weight")),
        packaging_type=_normalize_text(row.get("packaging_type")),
        type=_normalize_text(row.get("type")),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        source="batch_vlm",
        notes=_normalize_text(row.get("notes")),
    )


def barcode_baseline_groups() -> list[ImageGroup]:
    payloads = load_hackathon_image_payloads()
    by_barcode: dict[str, list] = defaultdict(list)
    singles: list = []
    for payload in payloads:
        validation = validate_barcode(extract_barcode(payload.image_bytes))
        if validation.is_valid:
            by_barcode[validation.value].append(payload)
        else:
            singles.append(payload)

    groups = [ImageGroup(group_id=f"barcode-{barcode}", images=sorted(images, key=lambda item: item.filename)) for barcode, images in sorted(by_barcode.items())]
    groups.extend(ImageGroup(group_id=f"single-{index:03d}", images=[payload]) for index, payload in enumerate(sorted(singles, key=lambda item: item.filename), start=1))
    return groups


async def model_groups(model_key: str, *, batch_size: int, concurrency: int) -> list[ImageGroup]:
    payloads = load_hackathon_image_payloads()
    client = get_vlm_client(model_key)
    preprocessed_payloads = [
        (payload, (payload.filename, preprocess(payload.image_bytes)))
        for payload in payloads
    ]
    evidence: list[ImageEvidence] = []
    batches = _chunked(preprocessed_payloads, batch_size)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_batch(index: int, batch_items: list[tuple[Any, tuple[str, bytes]]]) -> list[ImageEvidence]:
        async with semaphore:
            batch_id = f"{model_key}-batch-{index:03d}"
            print(f"[{model_key}] batch {index}/{len(batches)} start", file=sys.stderr, flush=True)
            rows = await _extract_batch_rows(client, [item for _, item in batch_items], batch_id)
            row_by_filename = {str(row.get("filename", "")).strip(): row for row in rows if isinstance(row, dict)}
            result: list[ImageEvidence] = []
            for payload, _ in batch_items:
                row = row_by_filename.get(payload.filename)
                if row is None:
                    row = {"filename": payload.filename}
                result.append(_evidence_from_row(payload, row))
            print(f"[{model_key}] batch {index}/{len(batches)} done", file=sys.stderr, flush=True)
            return result

    batch_results = await asyncio.gather(*(run_batch(index, batch_items) for index, batch_items in enumerate(batches, start=1)))
    for rows in batch_results:
        evidence.extend(rows)
    evidence_by_payload_id = {item.payload_id: item for item in evidence}
    return [cluster.to_image_group() for cluster in infer_product_groups(payloads, evidence_by_payload_id)]


async def model_groups_per_image(model_key: str, *, concurrency: int) -> list[ImageGroup]:
    payloads = load_hackathon_image_payloads()
    pipeline = ExtractionPipeline(client=get_vlm_client(model_key))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0

    async def run_one(payload):
        nonlocal completed
        async with semaphore:
            evidence = await pipeline.analyze_image_for_grouping(payload)
            completed += 1
            print(f"[{model_key}] image {completed}/{len(payloads)} done", file=sys.stderr, flush=True)
            return ImageEvidence(
                payload_id=payload.payload_id,
                filename=evidence.filename,
                image_hash=evidence.image_hash,
                barcode=evidence.barcode,
                barcode_is_valid=evidence.barcode_is_valid,
                barcode_type=evidence.barcode_type,
                item_name=evidence.item_name,
                brand=evidence.brand,
                weight=evidence.weight,
                packaging_type=evidence.packaging_type,
                type=evidence.type,
                confidence=evidence.confidence,
                source=evidence.source,
                notes=evidence.notes,
            )

    evidence = await asyncio.gather(*(run_one(payload) for payload in payloads))
    evidence_by_payload_id = {item.payload_id: item for item in evidence}
    return [cluster.to_image_group() for cluster in infer_product_groups(payloads, evidence_by_payload_id)]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--mode", choices=["per-image", "batched"], default="per-image")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    payloads = load_hackathon_image_payloads()
    truth = truth_groups_from_filenames(payloads)
    rows = []

    oracle = evaluate_group_predictions(truth, truth)
    rows.append(_report_row("filename_prefix_oracle", oracle))

    barcode = evaluate_group_predictions(truth, barcode_baseline_groups())
    rows.append(_report_row("barcode_baseline", barcode))

    for model_key in args.model_key:
        started = time.perf_counter()
        try:
            if args.mode == "batched":
                predicted = await model_groups(model_key, batch_size=args.batch_size, concurrency=args.concurrency)
            else:
                predicted = await model_groups_per_image(model_key, concurrency=args.concurrency)
            report = evaluate_group_predictions(truth, predicted)
            row = _report_row(model_key, report)
            row["runtime_seconds"] = round(time.perf_counter() - started, 2)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "name": model_key,
                    "error": str(exc),
                    "runtime_seconds": round(time.perf_counter() - started, 2),
                }
            )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
