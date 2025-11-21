from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image  # type: ignore

from images_mcp import convert_image, images_to_pdf


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "_out"


SUPPORTED_OUTPUT_FORMATS = [
    ("jpeg", ".jpg"),
    ("png", ".png"),
    ("webp", ".webp"),
    ("avif", ".avif"),
]


def example_images() -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    images: list[Path] = []
    for path in EXAMPLES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
            images.append(path)
    return images


def example_image_params():
    imgs = example_images()
    if not imgs:
        return [
            pytest.param(
                None, marks=pytest.mark.skip(reason="no examples/ images found")
            )
        ]
    return [pytest.param(p) for p in imgs]


@pytest.mark.parametrize("target_format,target_ext", SUPPORTED_OUTPUT_FORMATS)
@pytest.mark.parametrize("img_path", example_image_params())
def test_convert_examples(img_path, target_format, target_ext) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if img_path is None:
        pytest.skip("no examples/ images found")
    src = Path(img_path)
    out_file = OUTPUT_DIR / f"{src.stem}{target_ext}"

    if out_file.exists():
        out_file.unlink()

    result = convert_image(src, output_path=out_file, format=target_format, overwrite=True, quality=80)

    assert result.exists()
    assert result == out_file

    # Verify Pillow can open the output
    with Image.open(out_file) as im:
        im.verify()  # quick structural check


@pytest.mark.parametrize("img_path", example_image_params())
def test_derive_output_from_format(img_path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if img_path is None:
        pytest.skip("no examples/ images found")
    src = Path(img_path)

    result = convert_image(src, format="png", overwrite=True)

    assert result.exists()
    assert result.suffix.lower() == ".png"

    # Clean up derived file to avoid polluting repo
    result.unlink(missing_ok=True)


def test_images_to_pdf_merge() -> None:
    images = example_images()
    if len(images) < 2:
        pytest.skip("Need at least two example images to test PDF merge")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / "merged.pdf"
    if out_pdf.exists():
        out_pdf.unlink()

    result = images_to_pdf(input_paths=images[:3], output_pdf_path=out_pdf, overwrite=True)
    assert result.exists()
    assert result.suffix.lower() == ".pdf"
