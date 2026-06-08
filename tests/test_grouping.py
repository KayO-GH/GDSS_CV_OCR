from __future__ import annotations

from imdb_app.grouping import ImagePayload, group_images, group_key


def test_group_key_uses_prefix_before_underscore():
    assert group_key("S221234199_550719011.jpg") == "S221234199"
    assert group_key("single.jpg") == "single"


def test_group_images_sorts_groups_and_files():
    groups = group_images(
        [
            ImagePayload(filename="S2_2.jpg", image_bytes=b"2"),
            ImagePayload(filename="S1_1.jpg", image_bytes=b"1"),
            ImagePayload(filename="S2_1.jpg", image_bytes=b"3"),
        ]
    )

    assert [group.group_id for group in groups] == ["S1", "S2"]
    assert groups[1].filenames == ["S2_1.jpg", "S2_2.jpg"]
