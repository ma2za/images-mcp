from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, cast

from mcp.server.fastmcp import FastMCP
from PIL import Image, UnidentifiedImageError  # type: ignore
from unstructured.partition.auto import partition  # type: ignore
from openai import OpenAI  
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_file(path: str | Path) -> str:
    """Extract concatenated plain text from a document or image using Unstructured.

    The function uses `unstructured.partition.auto.partition` to parse the file and
    joins the textual representation of elements with double newlines.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    elements = partition(filename=str(file_path))
    return "\n\n".join(str(el) for el in elements)


_EXTENSION_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".avif": "AVIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".ico": "ICO",
}


def _infer_format_from_extension(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    return _EXTENSION_TO_FORMAT.get(ext)


def _derive_output_path(
    input_path: Path,
    requested_format: Optional[str],
    output_path: Optional[Path],
) -> Tuple[Path, str]:
    """
    Determine the final output path and Pillow format string.
    - If output_path provided: use its extension to infer format unless requested_format given.
    - Else: use input stem + requested_format (required) to build output path.
    """
    if output_path:
        output_format = requested_format or _infer_format_from_extension(output_path)
        if not output_format:
            raise ValueError(
                f"Cannot infer output format from extension '{output_path.suffix}'. Provide --format."
            )
        return output_path, output_format.upper()

    if not requested_format:
        raise ValueError(
            "Either output path or format must be provided to determine target file."
        )

    fmt_upper = requested_format.upper()
    # Find a representative extension for the target format; default to lowercase format
    ext = None
    for k_ext, k_fmt in _EXTENSION_TO_FORMAT.items():
        if k_fmt == fmt_upper:
            ext = k_ext
            break
    if ext is None:
        ext = f".{requested_format.lower()}"

    derived = input_path.with_suffix(ext)
    return derived, fmt_upper


def convert_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    format: Optional[str] = None,
    overwrite: bool = False,
    quality: Optional[int] = None,
    optimize: bool = True,
    progressive: Optional[bool] = None,
    keep_exif: bool = False,
) -> Path:
    """Convert an image to another format and write it to disk.

    Parameters
    ----------
    input_path: Path to the source image.
    output_path: Destination file path. If omitted, derived from input and format.
    format: Target format (e.g., "jpeg", "png", "webp"). Inferred from output extension if not provided.
    overwrite: Allow overwriting an existing output file.
    quality: Quality setting (JPEG/WEBP). 1-95 typically.
    optimize: Ask Pillow to optimize the output when supported.
    progressive: Use progressive encoding when supported (e.g., JPEG).
    keep_exif: Preserve EXIF metadata when available and supported by format.

    Returns
    -------
    Path to the written output file.
    """
    src = Path(input_path)
    if output_path is not None:
        out = Path(output_path)
    else:
        out = None

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    final_out, final_format = _derive_output_path(src, format, out)

    if final_out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use overwrite=True): {final_out}"
        )

    final_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as im:
            save_kwargs: dict = {}

            # Convert mode for formats that don't support alpha
            if final_format == "JPEG":
                if im.mode in ("RGBA", "LA"):
                    im = im.convert("RGB")
                elif im.mode == "P":
                    im = im.convert("RGB")

            if quality is not None:
                save_kwargs["quality"] = int(quality)

            if progressive is not None and final_format == "JPEG":
                save_kwargs["progressive"] = bool(progressive)

            # Optimize where supported
            save_kwargs["optimize"] = bool(optimize)

            # AVIF specifics: Pillow-AVIF uses quality and optional "effort"; keep simple
            if final_format == "AVIF":
                # AVIF requires RGB if input is palette
                if im.mode == "P":
                    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")

            # Preserve EXIF if requested and available
            if keep_exif and hasattr(im, "info") and "exif" in im.info:
                save_kwargs["exif"] = im.info["exif"]

            im.save(final_out, final_format, **save_kwargs)

    except UnidentifiedImageError as e:
        raise ValueError(f"Unrecognized image file: {src}") from e

    return final_out


def images_to_pdf(
    input_paths: Iterable[str | Path],
    output_pdf_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Merge a list of images into a single multi-page PDF.

    Notes:
        - All images are converted to RGB.
        - The order of pages follows the order of input_paths.
    """
    output_path = Path(output_pdf_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use overwrite=True): {output_path}"
        )

    image_files = [Path(p) for p in input_paths]
    if not image_files:
        raise ValueError("No input images provided for PDF conversion.")
    for p in image_files:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Open all images and convert to RGB
    opened_images: list[Image.Image] = []
    processed_images: list[Image.Image] = []
    try:
        for p in image_files:
            im_file = Image.open(p)
            # Convert to Image.Image type (convert() returns Image, not ImageFile)
            im: Image.Image = im_file.convert("RGB") if im_file.mode != "RGB" else im_file  # type: ignore[assignment]
            opened_images.append(im)

        # Normalize widths to ensure all PDF pages have the same width.
        # Use the smallest width to avoid upscaling; preserve aspect ratio.
        target_width = min(im.width for im in opened_images)
        for im in opened_images:
            print(f"Original width: {im.width}, Original height: {im.height}")
            if im.width != target_width:
                new_height = round(im.height * (target_width / im.width))
                resized = im.resize((target_width, new_height), resample=Image.Resampling.LANCZOS)
                print(f"Resized width: {resized.width}, Resized height: {resized.height}")
                processed_images.append(resized)
            else:
                processed_images.append(im)

        first, *rest = processed_images
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first.save(output_path, "PDF", save_all=True, append_images=rest)
    except UnidentifiedImageError as e:
        raise ValueError("One or more input files are not recognized as images.") from e
    finally:
        # Close both the originals and any resized images.
        for im in opened_images:
            try:
                im.close()
            except Exception:
                pass
        for im in processed_images:
            try:
                im.close()
            except Exception:
                pass

    return output_path


def _encode_image_to_base64(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _get_image_format_from_path(path: Path) -> str:
    """Get MIME type from file extension."""
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    else:
        return "image/png"  # default


mcp = FastMCP("image-conversion")


@mcp.tool()
async def convert(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    format: Optional[str] = None,
    overwrite: bool = False,
    quality: Optional[int] = None,
    optimize: bool = True,
    progressive: Optional[bool] = None,
    keep_exif: bool = False,
) -> str:
    result = convert_image(
        input_path=Path(input_path),
        output_path=Path(output_path) if output_path is not None else None,
        format=format,
        overwrite=overwrite,
        quality=quality,
        optimize=optimize,
        progressive=progressive,
        keep_exif=keep_exif,
    )
    return str(result.resolve())


@mcp.tool()
async def merge_pdf(
    output_pdf_path: str,
    input_paths: list[str],
    *,
    overwrite: bool = False,
) -> str:
    """Merge images into a single PDF.

    Args:
        output_pdf_path: Destination PDF file path.
        input_paths: Ordered list of image file paths.
        overwrite: Overwrite existing output.
    """
    result = images_to_pdf(
        input_paths=[Path(p) for p in input_paths],
        output_pdf_path=Path(output_pdf_path),
        overwrite=overwrite,
    )
    return str(result.resolve())


@mcp.tool()
async def extract_text(input_path: str) -> str:
    """Extract text from a file (images, PDFs, docs) using Unstructured."""
    return extract_text_from_file(Path(input_path))


@mcp.tool()
async def generate_image(
    prompt: str,
    output_path: str,
    *,
    model: str = "gpt-5",
    quality: Optional[str] = None,
    background: Optional[str] = None,
    input_images: Optional[list[str]] = None,
    mask_path: Optional[str] = None,
    input_fidelity: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    n: int = 1,
    overwrite: bool = False,
) -> str:
    """Generate or edit images using OpenAI's image generation API.
    
    This tool uses the Responses API to generate images from text prompts or edit
    existing images. It supports various customization options including quality
    and transparency.
    
    Args:
        prompt: Text prompt describing the image to generate or edit.
        output_path: Absolute path where the generated image(s) will be saved. 
            MUST be an absolute path. If n > 1, images will be saved as 
            output_path_0.png, output_path_1.png, etc.
        model: Model to use for generation. Defaults to "gpt-5". Supports gpt-4.1,
            gpt-5, and other models that support image generation.
        quality: Rendering quality. Options: "low", "medium", "high", or "auto" (default).
        background: Background type. Options: "transparent" or "opaque" (default).
            Only supported with png and webp formats.
        input_images: Optional list of absolute input image file paths to use as 
            references for editing or generating new images based on existing ones.
            All paths MUST be absolute.
        mask_path: Optional absolute path to a mask image for inpainting (editing 
            specific parts of an image). MUST be an absolute path. The mask must 
            have an alpha channel and match the size of the first input image.
        input_fidelity: Input fidelity level when using input images. Options:
            "low" (default) or "high" (better preserves details from input images).
        previous_response_id: Optional response ID from a previous call to continue
            a multi-turn conversation. When provided, the API will use the previous
            response as context for generating the new image.
        n: Number of images to generate (1-10). Defaults to 1.
        overwrite: Allow overwriting existing output files.
    
    Returns:
        JSON string containing:
        - "path": Absolute path to the generated image file (or first image if n > 1)
        - "response_id": Response ID from the API call, which can be used as 
          previous_response_id in follow-up calls for multi-turn conversations
    
    Raises:
        ValueError: If any path parameter is not an absolute path.
    
    Examples:
        # Generate a simple image (paths must be absolute)
        generate_image(
            "A gray tabby cat hugging an otter", 
            "/path/to/output.png"
        )
        
        # Generate with high quality and transparent background
        generate_image(
            "A 2D pixel art sprite sheet of a cat",
            "/path/to/sprite.png",
            quality="high",
            background="transparent"
        )
        
        # Edit an existing image
        generate_image(
            "Add a logo to the woman's top",
            "/path/to/edited.png",
            input_images=["/path/to/woman.jpg", "/path/to/logo.png"],
            input_fidelity="high"
        )
        
        # Multi-turn conversation (follow-up request)
        # First call returns JSON with path and response_id
        import json
        result1 = json.loads(generate_image(
            "Generate an image of a gray tabby cat hugging an otter",
            "/path/to/cat_otter.png"
        ))
        # Use the response_id for follow-up
        result2 = json.loads(generate_image(
            "Now make it look realistic",
            "/path/to/realistic.png",
            previous_response_id=result1["response_id"]
        ))
    """
    print(f"[generate_image] Starting image generation with prompt: '{prompt[:50]}...'")
    print(f"[generate_image] Output path: {output_path}, Model: {model}, Count: {n}")
    
    # Validate that all paths are absolute
    output_file = Path(output_path)
    if not output_file.is_absolute():
        error_msg = f"output_path must be an absolute path, got: {output_path}"
        print(f"[generate_image] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    if input_images:
        for img_path in input_images:
            img_file = Path(img_path)
            if not img_file.is_absolute():
                error_msg = f"All input_images paths must be absolute, got: {img_path}"
                print(f"[generate_image] ERROR: {error_msg}")
                raise ValueError(error_msg)
    
    if mask_path:
        mask_file = Path(mask_path)
        if not mask_file.is_absolute():
            error_msg = f"mask_path must be an absolute path, got: {mask_path}"
            print(f"[generate_image] ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    # Check if output exists
    if output_file.exists() and not overwrite:
        error_msg = f"Output file already exists (use overwrite=True): {output_file}"
        print(f"[generate_image] ERROR: {error_msg}")
        raise FileExistsError(error_msg)
    
    # Ensure output directory exists
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[generate_image] Output directory ready: {output_file.parent}")
    except OSError as e:
        error_msg = f"Failed to create output directory: {output_file.parent}"
        print(f"[generate_image] ERROR: {error_msg}: {e}")
        raise RuntimeError(error_msg) from e
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OPENAI_API_KEY environment variable is not set. Please set it to use image generation."
        print(f"[generate_image] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    try:
        client = OpenAI(api_key=api_key)
        print("[generate_image] OpenAI client initialized")
    except Exception as e:
        error_msg = f"Failed to initialize OpenAI client: {e}"
        print(f"[generate_image] ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e
    
    # Build input content
    print("[generate_image] Building input content...")
    input_content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    
    # Add input images if provided
    if input_images:
        print(f"[generate_image] Processing {len(input_images)} input image(s)...")
        for idx, img_path in enumerate(input_images):
            try:
                img_file = Path(img_path)
                if not img_file.exists():
                    error_msg = f"Input image not found: {img_file}"
                    print(f"[generate_image] ERROR: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                print(f"[generate_image] Encoding image {idx + 1}/{len(input_images)}: {img_file.name}")
                # Encode image to base64
                base64_image = _encode_image_to_base64(img_file)
                mime_type = _get_image_format_from_path(img_file)
                input_content.append({
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64_image}",
                })
                print(f"[generate_image] Image {idx + 1} encoded successfully")
            except Exception as e:
                error_msg = f"Failed to process input image {img_path}: {e}"
                print(f"[generate_image] ERROR: {error_msg}")
                raise RuntimeError(error_msg) from e
    
    # Build tool configuration
    print("[generate_image] Building tool configuration...")
    tool_config: dict = {"type": "image_generation"}
    
    if quality:
        tool_config["quality"] = quality
        print(f"[generate_image] Quality set to: {quality}")
    if background:
        tool_config["background"] = background
        print(f"[generate_image] Background set to: {background}")
    if input_fidelity:
        tool_config["input_fidelity"] = input_fidelity
        print(f"[generate_image] Input fidelity set to: {input_fidelity}")
    
    # Add mask if provided
    if mask_path:
        print(f"[generate_image] Processing mask: {mask_path}")
        try:
            # Check if mask_path is a file_id (starts with "file-") or a file path
            if mask_path.startswith("file-"):
                # Use file_id directly
                tool_config["input_image_mask"] = {"file_id": mask_path}
                print(f"[generate_image] Using mask file_id: {mask_path}")
            else:
                # Treat as file path - upload to Files API to get file_id
                mask_file = Path(mask_path)
                if not mask_file.exists():
                    error_msg = f"Mask image not found: {mask_file}"
                    print(f"[generate_image] ERROR: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                print(f"[generate_image] Uploading mask to Files API...")
                # Upload mask to Files API
                try:
                    with open(mask_file, "rb") as f:
                        mask_upload = client.files.create(
                            file=f,
                            purpose="vision"
                        )
                    tool_config["input_image_mask"] = {"file_id": mask_upload.id}
                    print(f"[generate_image] Mask uploaded successfully, file_id: {mask_upload.id}")
                except Exception as e:
                    error_msg = f"Failed to upload mask to Files API: {e}"
                    print(f"[generate_image] ERROR: {error_msg}")
                    raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to process mask: {e}"
            print(f"[generate_image] ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    # Generate images (handle multiple images)
    print(f"[generate_image] Starting generation of {n} image(s)...")
    generated_paths = []
    last_response_id: Optional[str] = None
    
    for i in range(n):
        try:
            print(f"[generate_image] Generating image {i + 1}/{n}...")
            
            # Construct input message - type checker has difficulty with Responses API structure
            # The Responses API accepts flexible input structures that don't match strict TypedDict types
            input_message: Any = {
                "role": "user",
                "content": input_content,
            }
            
            # Suppress type checking for Responses API call due to complex union types
            print(f"[generate_image] Calling OpenAI Responses API with model: {model}")
            try:
                api_kwargs: dict[str, Any] = {
                    "model": model,
                    "input": [input_message],  # type: ignore[list-item]
                    "tools": [tool_config],  # type: ignore[list-item]
                }
                if previous_response_id:
                    api_kwargs["previous_response_id"] = previous_response_id
                    print(f"[generate_image] Using previous_response_id: {previous_response_id}")
                
                response = client.responses.create(**api_kwargs)  # type: ignore[call-overload,arg-type]
                last_response_id = response.id
                print(f"[generate_image] API call successful, response received (ID: {last_response_id})")
            except Exception as api_error:
                error_msg = f"OpenAI API call failed: {api_error}"
                print(f"[generate_image] ERROR: {error_msg}")
                # Re-raise with more context
                if hasattr(api_error, 'status_code'):
                    print(f"[generate_image] API status code: {api_error.status_code}")
                if hasattr(api_error, 'response'):
                    print(f"[generate_image] API response: {api_error.response}")
                raise RuntimeError(error_msg) from api_error
            
            # Extract image data from response
            print("[generate_image] Extracting image data from response...")
            image_data = [
                output.result
                for output in response.output
                if output.type == "image_generation_call" and output.result is not None
            ]
            
            if not image_data:
                # Check if there's an error message
                error_msg = getattr(response.output[0], "content", "Unknown error") if response.output else "Unknown error"
                print(f"[generate_image] ERROR: No image data in response. Error: {error_msg}")
                raise ValueError(f"Image generation failed: {error_msg}")
            
            print(f"[generate_image] Image data extracted successfully ({len(image_data)} image(s))")
            
            # Determine output path for this image
            if n > 1:
                # Multiple images: add index to filename
                stem = output_file.stem
                suffix = output_file.suffix or ".png"
                current_output = output_file.parent / f"{stem}_{i}{suffix}"
            else:
                current_output = output_file
            
            print(f"[generate_image] Saving image to: {current_output}")
            
            # Decode and save image
            try:
                image_base64 = image_data[0]
                if image_base64 is None:
                    error_msg = "Generated image data is None"
                    print(f"[generate_image] ERROR: {error_msg}")
                    raise ValueError(error_msg)
                
                print("[generate_image] Decoding base64 image data...")
                image_bytes = base64.b64decode(image_base64)  # type: ignore[arg-type]
                print(f"[generate_image] Image decoded, size: {len(image_bytes)} bytes")
                
                with open(current_output, "wb") as f:
                    f.write(image_bytes)
                print(f"[generate_image] Image saved successfully: {current_output}")
                
                generated_paths.append(str(current_output.resolve()))
                print(f"[generate_image] Image {i + 1}/{n} completed successfully")
                
            except Exception as file_error:
                error_msg = f"Failed to save image to {current_output}: {file_error}"
                print(f"[generate_image] ERROR: {error_msg}")
                raise RuntimeError(error_msg) from file_error
            
        except ValueError as e:
            # Re-raise ValueError as-is (these are expected errors)
            print(f"[generate_image] ValueError: {e}")
            raise
        except RuntimeError as e:
            # Re-raise RuntimeError as-is (these are wrapped errors)
            raise
        except Exception as e:
            error_msg = f"Unexpected error generating image {i + 1}/{n}: {str(e)}"
            print(f"[generate_image] ERROR: {error_msg}")
            print(f"[generate_image] Exception type: {type(e).__name__}")
            raise RuntimeError(error_msg) from e
    
    result_path = generated_paths[0]
    
    # Return JSON with path and response_id for multi-turn support
    result = {
        "path": result_path,
        "response_id": last_response_id,
    }
    result_json = json.dumps(result)
    print(f"[generate_image] All images generated successfully. Returning: {result_json}")
    return result_json


def main() -> None:
    # Run as stdio transport per MCP best practices for local servers
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
