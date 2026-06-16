from __future__ import annotations

from imdb_app.grouping import (
    ImageEvidence,
    ImagePayload,
    group_images,
    group_images_by_filename_prefix,
    group_key,
    infer_product_groups,
)


def test_group_key_uses_prefix_before_underscore():
    assert group_key("S221234199_550719011.jpg") == "S221234199"
    assert group_key("single.jpg") == "single"


def test_group_images_sorts_groups_and_files():
    groups = group_images_by_filename_prefix(
        [
            ImagePayload(filename="S2_2.jpg", image_bytes=b"2"),
            ImagePayload(filename="S1_1.jpg", image_bytes=b"1"),
            ImagePayload(filename="S2_1.jpg", image_bytes=b"3"),
        ]
    )

    assert [group.group_id for group in groups] == ["S1", "S2"]
    assert groups[1].filenames == ["S2_1.jpg", "S2_2.jpg"]


def test_group_images_alias_preserves_prefix_grouping_for_legacy_callers():
    groups = group_images(
        [
            ImagePayload(filename="S1_2.jpg", image_bytes=b"2"),
            ImagePayload(filename="S1_1.jpg", image_bytes=b"1"),
        ]
    )

    assert [group.group_id for group in groups] == ["S1"]


def test_infer_product_groups_uses_valid_barcode_not_filename():
    first = ImagePayload(filename="random-front.jpg", image_bytes=b"1")
    second = ImagePayload(filename="other-back.jpg", image_bytes=b"2")
    images = [first, second]
    evidence = {
        first.payload_id: ImageEvidence(
            payload_id=first.payload_id, filename="random-front.jpg", image_hash="h1", barcode="6034000482027", barcode_is_valid=True
        ),
        second.payload_id: ImageEvidence(
            payload_id=second.payload_id, filename="other-back.jpg", image_hash="h2", barcode="6034000482027", barcode_is_valid=True
        ),
    }

    clusters = infer_product_groups(images, evidence)

    assert [cluster.group_id for cluster in clusters] == ["barcode-6034000482027"]
    assert sorted(clusters[0].filenames) == ["other-back.jpg", "random-front.jpg"]


def test_infer_product_groups_keeps_conflicting_barcodes_separate_even_with_same_prefix():
    first = ImagePayload(filename="same_1.jpg", image_bytes=b"1")
    second = ImagePayload(filename="same_2.jpg", image_bytes=b"2")
    images = [first, second]
    evidence = {
        first.payload_id: ImageEvidence(
            payload_id=first.payload_id, filename="same_1.jpg", image_hash="h1", barcode="6034000482027", barcode_is_valid=True
        ),
        second.payload_id: ImageEvidence(
            payload_id=second.payload_id, filename="same_2.jpg", image_hash="h2", barcode="8410300363439", barcode_is_valid=True
        ),
    }

    clusters = infer_product_groups(images, evidence)

    assert [cluster.group_id for cluster in clusters] == ["barcode-6034000482027", "barcode-8410300363439"]


def test_infer_product_groups_clusters_strong_non_barcode_evidence():
    first = ImagePayload(filename="front.png", image_bytes=b"1")
    second = ImagePayload(filename="back.png", image_bytes=b"2")
    images = [first, second]
    evidence = {
        first.payload_id: ImageEvidence(
            payload_id=first.payload_id,
            filename="front.png",
            image_hash="h1",
            brand="BAMA",
            item_name="BAMA MAYONNAISE WITH LEMON",
            weight="909ML",
            packaging_type="GLASS JAR",
            type="MAYONNAISE",
            confidence=0.8,
        ),
        second.payload_id: ImageEvidence(
            payload_id=second.payload_id,
            filename="back.png",
            image_hash="h2",
            brand="BAMA",
            item_name="BAMA MAYONNAISE WITH A DASH OF LEMON",
            weight="909ML",
            packaging_type="GLASS JAR",
            type="MAYONNAISE",
            confidence=0.75,
        ),
    }

    clusters = infer_product_groups(images, evidence)

    assert len(clusters) == 1
    assert clusters[0].group_id == "auto-001"
    assert clusters[0].needs_review is False


def test_infer_product_groups_separates_similar_names_with_different_weight():
    first = ImagePayload(filename="small.png", image_bytes=b"1")
    second = ImagePayload(filename="large.png", image_bytes=b"2")
    images = [first, second]
    evidence = {
        first.payload_id: ImageEvidence(
            payload_id=first.payload_id,
            filename="small.png",
            image_hash="h1",
            brand="POMO",
            item_name="POMO TOMATO MIX",
            weight="60G",
            type="TOMATO MIX",
            confidence=0.8,
        ),
        second.payload_id: ImageEvidence(
            payload_id=second.payload_id,
            filename="large.png",
            image_hash="h2",
            brand="POMO",
            item_name="POMO TOMATO MIX",
            weight="380G",
            type="TOMATO MIX",
            confidence=0.8,
        ),
    }

    clusters = infer_product_groups(images, evidence)

    assert len(clusters) == 2
    assert all(cluster.needs_review for cluster in clusters)


def test_infer_product_groups_keeps_duplicate_filenames_distinct():
    first = ImagePayload(filename="IMG_0001.jpg", image_bytes=b"1")
    second = ImagePayload(filename="IMG_0001.jpg", image_bytes=b"2")
    images = [first, second]
    evidence = {
        first.payload_id: ImageEvidence(
            payload_id=first.payload_id,
            filename="IMG_0001.jpg",
            image_hash="h1",
            barcode="6034000482027",
            barcode_is_valid=True,
        ),
        second.payload_id: ImageEvidence(
            payload_id=second.payload_id,
            filename="IMG_0001.jpg",
            image_hash="h2",
            barcode="8410300363439",
            barcode_is_valid=True,
        ),
    }

    clusters = infer_product_groups(images, evidence)

    assert [cluster.group_id for cluster in clusters] == ["barcode-6034000482027", "barcode-8410300363439"]
