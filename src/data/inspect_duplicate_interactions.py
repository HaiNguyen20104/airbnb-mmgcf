import pandas as pd


REVIEWS_PATH = "data/raw/bangkok/reviews.csv.gz"


def main():
    reviews = pd.read_csv(REVIEWS_PATH)

    duplicated = reviews[
        reviews.duplicated(
            subset=["reviewer_id", "listing_id"],
            keep=False
        )
    ].sort_values(["reviewer_id", "listing_id"])

    print("Number of rows involved in duplicated interactions:")
    print(len(duplicated))

    print()
    print("Number of unique duplicated User-Item pairs:")
    print(
        duplicated[
            ["reviewer_id", "listing_id"]
        ].drop_duplicates().shape[0]
    )

    print()
    print("Sample:")
    print(
        duplicated[
            [
                "id",
                "listing_id",
                "date",
                "reviewer_id",
                "reviewer_name",
                "comments",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()