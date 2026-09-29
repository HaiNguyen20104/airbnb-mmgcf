import pandas as pd


LISTINGS_PATH = "data/raw/bangkok/listings.csv.gz"
REVIEWS_PATH = "data/raw/bangkok/reviews.csv.gz"


def main():
    listings = pd.read_csv(LISTINGS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    # ==========================================
    # 1. PRICE
    # ==========================================

    print("=== PRICE ===")

    missing_price = listings["price"].isna().sum()

    print(f"Missing price: {missing_price}")

    print()
    print("Price data type:")
    print(listings["price"].dtype)

    print()
    print("Sample price values:")
    print(
        listings["price"]
        .head(20)
        .to_string(index=False)
    )

    # Convert price from string to numeric
    price_numeric = (
        listings["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    print()
    print("Numeric price statistics:")
    print(price_numeric.describe())

    # Check zero and negative prices
    invalid_price = (price_numeric <= 0).sum()
    negative_price = (price_numeric < 0).sum()
    zero_price = (price_numeric == 0).sum()

    print()
    print(f"Invalid price (<= 0): {invalid_price}")
    print(f"Zero price: {zero_price}")
    print(f"Negative price: {negative_price}")

    # ==========================================
    # 2. LOCATION
    # ==========================================

    print()
    print("=== LOCATION ===")

    latitude_invalid = (
        (listings["latitude"] < -90)
        | (listings["latitude"] > 90)
    ).sum()

    longitude_invalid = (
        (listings["longitude"] < -180)
        | (listings["longitude"] > 180)
    ).sum()

    print(f"Invalid latitude: {latitude_invalid}")
    print(f"Invalid longitude: {longitude_invalid}")

    print()
    print("Latitude range:")
    print(
        f"min={listings['latitude'].min()}, "
        f"max={listings['latitude'].max()}"
    )

    print()
    print("Longitude range:")
    print(
        f"min={listings['longitude'].min()}, "
        f"max={listings['longitude'].max()}"
    )

    # ==========================================
    # 3. PICTURE URL
    # ==========================================

    print()
    print("=== PICTURE URL ===")

    missing_picture = listings["picture_url"].isna().sum()

    empty_picture = (
        listings["picture_url"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Missing picture_url: {missing_picture}")
    print(f"Empty picture_url: {empty_picture}")

    # ==========================================
    # 4. INTERACTION KEYS
    # ==========================================

    print()
    print("=== INTERACTION KEYS ===")

    missing_reviewer_id = reviews["reviewer_id"].isna().sum()
    missing_listing_id = reviews["listing_id"].isna().sum()

    print(f"Missing reviewer_id: {missing_reviewer_id}")
    print(f"Missing listing_id: {missing_listing_id}")


if __name__ == "__main__":
    main()