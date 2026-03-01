from __future__ import annotations

from acp.schema import (
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from deepagents_acp.utils import (
    convert_embedded_resource_block_to_content_blocks,
    convert_image_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
)


def test_convert_text_block_to_content_blocks() -> None:
    out = convert_text_block_to_content_blocks(TextContentBlock(type="text", text="hi"))
    assert out == [{"type": "text", "text": "hi"}]


def test_convert_image_block_to_content_blocks_with_data() -> None:
    out = convert_image_block_to_content_blocks(
        ImageContentBlock(type="image", mime_type="image/png", data="AAAA")
    )
    assert out == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]


def test_convert_image_block_to_content_blocks_without_data_falls_back_to_text() -> None:
    out = convert_image_block_to_content_blocks(
        ImageContentBlock(type="image", mime_type="image/png", data="")
    )
    assert out == [{"type": "text", "text": "[Image: no data available]"}]


def test_convert_resource_block_to_content_blocks_truncates_root_dir() -> None:
    block = ResourceContentBlock(
        type="resource_link",
        name="file",
        uri="file:///root/subdir/file.txt",
        description=None,
        mime_type=None,
    )
    out = convert_resource_block_to_content_blocks(block, root_dir="/root")
    assert out == [{"type": "text", "text": "[Resource: file\nURI: file://subdir/file.txt]"}]


def test_convert_embedded_resource_block_to_content_blocks_text() -> None:
    block = EmbeddedResourceContentBlock(
        type="resource",
        resource=TextResourceContents(
            mime_type="text/plain",
            text="hello",
            uri="file:///mem.txt",
        ),
    )
    out = convert_embedded_resource_block_to_content_blocks(block)
    assert out == [{"type": "text", "text": "[Embedded text/plain resource: hello"}]
