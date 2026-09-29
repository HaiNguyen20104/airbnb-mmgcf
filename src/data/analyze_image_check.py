import pandas as pd


IMAGE_CHECK_PATH = "data/processed/bangkok/image_check.csv"


def main():
    results = pd.read_csv(IMAGE_CHECK_PATH)

    print("=== HTTP ERROR STATUS ===")

    http_errors = results[
        results["image_status"] == "http_error"
    ]

    print(
        http_errors["http_status"]
        .value_counts()
    )

    print()
    print("=== IMAGE STATUS ===")

    print(
        results["image_status"]
        .value_counts()
    )

    print()
    print("=== NON-VALID EXAMPLES ===")

    print(
        results[
            results["image_status"] != "valid"
        ][
            [
                "listing_id",
                "image_status",
                "http_status",
                "content_type",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()