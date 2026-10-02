import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MAPPING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "item_mapping.parquet"
)

LISTINGS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "listings_kcore.parquet"
)

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "images"
)

STATUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "image_download_status.parquet"
)


# ============================================================
# CONFIG
# ============================================================

# Number of concurrent download threads.
MAX_WORKERS = 4

# Number of listings processed in one scheduling batch.
BATCH_SIZE = 64

TIMEOUT = 30
MAX_RETRIES = 2

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0 Safari/537.36"
)


# One requests.Session per worker thread.
_thread_local = threading.local()


def get_session():
    """Return one reusable requests.Session for each worker thread."""
    session = getattr(_thread_local, "session", None)

    if session is None:
        session = requests.Session()

        retry = Retry(
            total=MAX_RETRIES,
            connect=MAX_RETRIES,
            read=MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=1,
            pool_maxsize=1,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update(
            {"User-Agent": USER_AGENT}
        )

        _thread_local.session = session

    return session


def validate_image(image_bytes):
    """
    Validate that response bytes contain a readable image.
    Returns image metadata without saving/keeping the image in memory
    after this function returns.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size

            # Force actual decoding to detect corrupted images.
            image.load()

            if width <= 0 or height <= 0:
                return (
                    False,
                    image_format,
                    width,
                    height,
                    "invalid_dimensions",
                )

            return (
                True,
                image_format,
                width,
                height,
                None,
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        return (
            False,
            None,
            None,
            None,
            str(exc),
        )


def extension_from_format(image_format):
    """Convert PIL image format to a file extension."""
    format_map = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }

    return format_map.get(
        str(image_format).upper(),
        ".img",
    )


def get_cached_image(item_idx):
    """
    Find an existing cached image for item_idx and validate it.
    Returns (status, path, format, width, height) or None.
    """
    for existing_file in IMAGE_DIR.glob(
        f"{item_idx}.*"
    ):
        try:
            with Image.open(existing_file) as image:
                image.load()

                width, height = image.size

                if width > 0 and height > 0:
                    return (
                        "cached",
                        existing_file,
                        image.format,
                        width,
                        height,
                    )

        except (
            UnidentifiedImageError,
            OSError,
        ):
            # Remove corrupt cached file.
            try:
                existing_file.unlink()
            except OSError:
                pass

    return None


def download_one(item_idx, listing_id, picture_url):
    """
    Download and validate one image.

    Returns:
        result dict
    """
    result = {
        "item_idx": item_idx,
        "listing_id": listing_id,
        "picture_url": picture_url,
        "status": None,
        "local_path": None,
        "http_status": None,
        "content_type": None,
        "image_format": None,
        "width": None,
        "height": None,
        "error": None,
    }

    if not picture_url:
        result["status"] = "missing_url"
        result["error"] = "picture_url is empty"
        return result

    # Reuse an existing valid image if available.
    cached = get_cached_image(item_idx)

    if cached is not None:
        (
            status,
            path,
            image_format,
            width,
            height,
        ) = cached

        result["status"] = status
        result["local_path"] = str(
            path.relative_to(BASE_DIR)
        )
        result["image_format"] = image_format
        result["width"] = width
        result["height"] = height

        return result

    session = get_session()

    try:
        response = session.get(
            picture_url,
            timeout=TIMEOUT,
        )

        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get(
            "Content-Type"
        )

        if response.status_code != 200:
            result["status"] = "http_error"
            result["error"] = (
                f"HTTP {response.status_code}"
            )
            return result

        image_bytes = response.content

        if not image_bytes:
            result["status"] = "empty_response"
            result["error"] = "Response body is empty"
            return result

        (
            is_valid,
            image_format,
            width,
            height,
            validation_error,
        ) = validate_image(image_bytes)

        if not is_valid:
            result["status"] = "invalid_image"
            result["image_format"] = image_format
            result["width"] = width
            result["height"] = height
            result["error"] = validation_error
            return result

        extension = extension_from_format(
            image_format
        )

        output_path = (
            IMAGE_DIR
            / f"{item_idx}{extension}"
        )

        # Use a temporary file so an interrupted write does
        # not leave a partially written image with final name.
        temp_path = (
            IMAGE_DIR
            / f".{item_idx}.tmp"
        )

        temp_path.write_bytes(image_bytes)

        # Path.replace() works on both Windows and macOS.
        temp_path.replace(output_path)

        result["status"] = "downloaded"
        result["local_path"] = str(
            output_path.relative_to(BASE_DIR)
        )
        result["image_format"] = image_format
        result["width"] = width
        result["height"] = height

        return result

    except requests.Timeout:
        result["status"] = "timeout"
        result["error"] = (
            f"Request timed out after {TIMEOUT}s"
        )
        return result

    except requests.RequestException as exc:
        result["status"] = "request_error"
        result["error"] = str(exc)
        return result

    except OSError as exc:
        result["status"] = "file_error"
        result["error"] = str(exc)
        return result


def process_batch(batch_df):
    """Download one batch using MAX_WORKERS threads."""
    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for row in batch_df.itertuples(
            index=False
        ):
            future = executor.submit(
                download_one,
                int(row.item_idx),
                str(row.listing_id),
                str(row.picture_url).strip(),
            )

            futures[future] = int(row.item_idx)

        for future in as_completed(futures):
            results.append(
                future.result()
            )

    return results


def main():
    print("=== IMAGE DOWNLOAD ===")

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    mapping = pd.read_parquet(
        MAPPING_PATH
    )

    listings = pd.read_parquet(
        LISTINGS_PATH
    )

    required_mapping_columns = {
        "item_idx",
        "listing_id",
    }

    required_listing_columns = {
        "id",
        "picture_url",
    }

    missing_mapping = (
        required_mapping_columns
        - set(mapping.columns)
    )

    missing_listings = (
        required_listing_columns
        - set(listings.columns)
    )

    if missing_mapping:
        raise ValueError(
            "Mapping is missing columns: "
            f"{sorted(missing_mapping)}"
        )

    if missing_listings:
        raise ValueError(
            "Listings are missing columns: "
            f"{sorted(missing_listings)}"
        )

    # --------------------------------------------------
    # 2. Merge mapping -> listings
    # --------------------------------------------------
    df = pd.merge(
        mapping,
        listings[["id", "picture_url"]],
        left_on="listing_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(mapping):
        raise ValueError(
            "Merged data size does not match "
            "item mapping."
        )

    if df["picture_url"].isna().any():
        missing_count = int(
            df["picture_url"].isna().sum()
        )

        raise ValueError(
            f"{missing_count} mapped listings "
            "have no picture_url."
        )

    df = (
        df
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. Validate item_idx
    # --------------------------------------------------
    expected_item_idx = pd.Series(
        range(len(df)),
        dtype=df["item_idx"].dtype,
    )

    if not df["item_idx"].equals(
        expected_item_idx
    ):
        raise ValueError(
            "item_idx must be continuous "
            "from 0 to N-1."
        )

    if df["item_idx"].duplicated().any():
        raise ValueError(
            "Duplicate item_idx found."
        )

    if df["listing_id"].duplicated().any():
        raise ValueError(
            "Duplicate listing_id found."
        )

    # --------------------------------------------------
    # 4. Download in batches
    # --------------------------------------------------
    total_items = len(df)
    all_results = []

    total_batches = (
        total_items + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(f"Listings: {total_items}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image directory: {IMAGE_DIR}")
    print()

    for batch_number, start in enumerate(
        range(
            0,
            total_items,
            BATCH_SIZE,
        ),
        start=1,
    ):
        end = min(
            start + BATCH_SIZE,
            total_items,
        )

        batch_df = df.iloc[start:end]

        print(
            f"[Batch {batch_number}/{total_batches}] "
            f"items {start}-{end - 1}"
        )

        batch_results = process_batch(
            batch_df
        )

        # Restore item_idx order inside batch.
        batch_results.sort(
            key=lambda x: x["item_idx"]
        )

        all_results.extend(
            batch_results
        )

        downloaded = sum(
            r["status"] == "downloaded"
            for r in batch_results
        )

        cached = sum(
            r["status"] == "cached"
            for r in batch_results
        )

        failed = len(batch_results) - (
            downloaded + cached
        )

        print(
            f"  downloaded={downloaded}, "
            f"cached={cached}, "
            f"failed={failed}"
        )

    # --------------------------------------------------
    # 5. Restore global item_idx order
    # --------------------------------------------------
    status_df = (
        pd.DataFrame(all_results)
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 6. Validate status table
    # --------------------------------------------------
    if len(status_df) != len(mapping):
        raise ValueError(
            "Status row count does not match "
            "item mapping."
        )

    if status_df["item_idx"].duplicated().any():
        raise ValueError(
            "Duplicate item_idx found in "
            "download status."
        )

    if status_df["listing_id"].duplicated().any():
        raise ValueError(
            "Duplicate listing_id found in "
            "download status."
        )

    if not status_df["item_idx"].equals(
        mapping["item_idx"].reset_index(
            drop=True
        )
    ):
        raise ValueError(
            "Download status order does not "
            "match item_mapping order."
        )

    # --------------------------------------------------
    # 7. Save status
    # --------------------------------------------------
    status_df.to_parquet(
        STATUS_PATH,
        index=False,
    )

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------
    print()
    print("=== DOWNLOAD SUMMARY ===")

    counts = (
        status_df["status"]
        .value_counts()
    )

    for status, count in counts.items():
        print(f"{status}: {count}")

    success_count = int(
        status_df["status"]
        .isin({"downloaded", "cached"})
        .sum()
    )

    failed_count = (
        len(status_df) - success_count
    )

    print()
    print(
        f"Successful/cached images: "
        f"{success_count}"
    )
    print(
        f"Failed images: {failed_count}"
    )

    print()
    print(
        "Saved status:"
        f" {STATUS_PATH}"
    )

    if failed_count > 0:
        print()
        print(
            "WARNING: Some images were not "
            "downloaded successfully."
        )
        print(
            "No listings were removed."
        )
    else:
        print()
        print(
            "PASS: All mapped listings have "
            "valid local images."
        )


if __name__ == "__main__":
    main()
