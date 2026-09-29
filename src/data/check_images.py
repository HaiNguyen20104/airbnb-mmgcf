import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed


LISTINGS_PATH = "data/raw/bangkok/listings.csv.gz"
OUTPUT_PATH = "data/processed/bangkok/image_check.csv"

REQUEST_TIMEOUT = 10
MAX_WORKERS = 20


def check_image_url(row):
    listing_id = row["id"]
    url = row["picture_url"]

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        status_code = response.status_code
        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        response.close()

        if status_code != 200:
            return {
                "listing_id": listing_id,
                "picture_url": url,
                "image_status": "http_error",
                "http_status": status_code,
                "content_type": content_type,
            }

        if not content_type.lower().startswith("image/"):
            return {
                "listing_id": listing_id,
                "picture_url": url,
                "image_status": "not_image",
                "http_status": status_code,
                "content_type": content_type,
            }

        return {
            "listing_id": listing_id,
            "picture_url": url,
            "image_status": "valid",
            "http_status": status_code,
            "content_type": content_type,
        }

    except requests.exceptions.Timeout:
        return {
            "listing_id": listing_id,
            "picture_url": url,
            "image_status": "timeout",
            "http_status": None,
            "content_type": "",
        }

    except requests.exceptions.RequestException:
        return {
            "listing_id": listing_id,
            "picture_url": url,
            "image_status": "request_error",
            "http_status": None,
            "content_type": "",
        }

    except Exception:
        return {
            "listing_id": listing_id,
            "picture_url": url,
            "image_status": "unknown_error",
            "http_status": None,
            "content_type": "",
        }


def main():
    listings = pd.read_csv(LISTINGS_PATH)

    total = len(listings)

    print(f"Total listings: {total}")
    print(f"Checking with {MAX_WORKERS} workers...")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(check_image_url, row)
            for _, row in listings.iterrows()
        ]

        for index, future in enumerate(
            as_completed(futures),
            start=1
        ):
            results.append(future.result())

            if index % 100 == 0:
                print(f"Checked {index}/{total}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")

    print()
    print("=== IMAGE CHECK RESULT ===")

    print(
        results_df["image_status"]
        .value_counts()
    )

    print()
    print("=== CONTENT TYPE ===")

    print(
        results_df["content_type"]
        .value_counts()
        .head(20)
    )


if __name__ == "__main__":
    main()