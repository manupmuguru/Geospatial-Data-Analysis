# Geospatial-Data-Analysis
# ================================================================
#                 GEOSPATIAL DATA ANALYSIS
# ================================================================
#
# OBJECTIVE:
# Visualize sales data geographically and detect underserved
# regional markets.
#
# ANALYSIS:
# 1. Load regional dataset
# 2. Clean data
# 3. Handle missing values
# 4. Remove duplicates
# 5. Analyze State, City and Postal Code
# 6. Aggregate revenue by region
# 7. Calculate user/customer density
# 8. Calculate average revenue
# 9. Calculate demand indicators
# 10. Identify underserved regions
# 11. Identify top 3 high-potential regions
# 12. Create geographic scatter plot
# 13. Create interactive Folium map
# 14. Create marker clusters
# 15. Create heatmap
# 16. Perform K-Means geographic clustering
# 17. Generate summary tables
# 18. Export all analysis results
#
# DATASET:
# regional_map_dataset.csv
#
# ================================================================


# ================================================================
# 1. INSTALL REQUIRED LIBRARIES
# ================================================================
#
# Run these commands once in Command Prompt / Terminal if required:
#
# pip install pandas numpy matplotlib seaborn folium scikit-learn
#
# ================================================================


# ================================================================
# 2. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from folium.plugins import MarkerCluster
from folium.plugins import HeatMap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 150)


# ================================================================
# 3. PROJECT TITLE
# ================================================================

print("\n")
print("=" * 90)
print("                    GEOSPATIAL DATA ANALYSIS")
print("=" * 90)

print("""
Objective:
Visualize sales data geographically and detect underserved
regional markets with high potential for expansion.
""")


# ================================================================
# 4. LOAD DATASET
# ================================================================

FILE_NAME = "regional_map_dataset.csv"

if not os.path.exists(FILE_NAME):

    print("\nERROR:")
    print("regional_map_dataset.csv was not found.")
    print("Please place the dataset in the same folder as this program.")

    raise SystemExit


df = pd.read_csv(FILE_NAME)


print("\nDataset loaded successfully.")

print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


print("\nFirst 10 records:")
print(df.head(10))


# ================================================================
# 5. DATASET INFORMATION
# ================================================================

print("\n")
print("=" * 90)
print("DATASET INFORMATION")
print("=" * 90)


print("\nColumn names:")

for column in df.columns:

    print("-", column)


print("\nData types:")

print(df.dtypes)


print("\nMissing values:")

print(df.isnull().sum())


print("\nDuplicate records:")

print(df.duplicated().sum())


# ================================================================
# 6. CLEAN COLUMN NAMES
# ================================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)


print("\nCleaned column names:")

print(df.columns.tolist())


# ================================================================
# 7. REMOVE DUPLICATES
# ================================================================

duplicate_count = df.duplicated().sum()


if duplicate_count > 0:

    df = df.drop_duplicates()


print("\nDuplicate records removed:", duplicate_count)

print("Remaining rows:", len(df))


# ================================================================
# 8. FUNCTION TO FIND COLUMN
# ================================================================

def find_column(possible_names):

    for name in possible_names:

        name = name.lower()

        if name in df.columns:

            return name

    return None


# ================================================================
# 9. IDENTIFY LOCATION COLUMNS
# ================================================================

state_col = find_column([
    "state",
    "state_name",
    "region",
    "province"
])


city_col = find_column([
    "city",
    "city_name",
    "town"
])


postal_col = find_column([
    "postal_code",
    "postcode",
    "zip_code",
    "zipcode",
    "zip"
])


latitude_col = find_column([
    "latitude",
    "lat",
    "y"
])


longitude_col = find_column([
    "longitude",
    "lon",
    "lng",
    "long",
    "x"
])


# ================================================================
# 10. IDENTIFY BUSINESS COLUMNS
# ================================================================

revenue_col = find_column([
    "revenue",
    "sales",
    "sales_amount",
    "total_sales",
    "total_revenue",
    "amount"
])


user_col = find_column([
    "users",
    "user_count",
    "customers",
    "customer_count",
    "users_count",
    "population"
])


demand_col = find_column([
    "demand",
    "demand_score",
    "orders",
    "order_count",
    "transactions"
])


# ================================================================
# 11. PRINT DETECTED COLUMNS
# ================================================================

print("\n")
print("=" * 90)
print("DETECTED COLUMNS")
print("=" * 90)

print("State column       :", state_col)
print("City column        :", city_col)
print("Postal column      :", postal_col)
print("Latitude column    :", latitude_col)
print("Longitude column   :", longitude_col)
print("Revenue column     :", revenue_col)
print("User column        :", user_col)
print("Demand column      :", demand_col)


# ================================================================
# 12. CHECK REQUIRED LOCATION DATA
# ================================================================

if latitude_col is None or longitude_col is None:

    print("\nWARNING:")
    print("Latitude/Longitude columns were not detected.")

    print("""
The dataset should ideally contain:
    latitude
    longitude

Example:
    latitude = 12.9716
    longitude = 77.5946
""")


# ================================================================
# 13. HANDLE MISSING VALUES
# ================================================================

print("\n")
print("=" * 90)
print("DATA CLEANING")
print("=" * 90)


# Numeric columns

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Text columns

categorical_columns = df.select_dtypes(
    include="object"
).columns


for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if len(mode_value) > 0:

            df[column] = df[column].fillna(
                mode_value[0]
            )


print("Missing-value treatment completed.")


# ================================================================
# 14. CONVERT NUMERIC COLUMNS
# ================================================================

numeric_target_columns = [
    revenue_col,
    user_col,
    demand_col,
    latitude_col,
    longitude_col
]


for column in numeric_target_columns:

    if column is not None:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ================================================================
# 15. REMOVE INVALID COORDINATES
# ================================================================

if latitude_col is not None and longitude_col is not None:

    df = df.dropna(
        subset=[
            latitude_col,
            longitude_col
        ]
    )


    df = df[
        (df[latitude_col] >= -90) &
        (df[latitude_col] <= 90) &
        (df[longitude_col] >= -180) &
        (df[longitude_col] <= 180)
    ]


print("\nValid geographic records:", len(df))


# ================================================================
# 16. BASIC STATISTICS
# ================================================================

print("\n")
print("=" * 90)
print("BASIC STATISTICS")
print("=" * 90)


if revenue_col is not None:

    print("\nRevenue statistics:")

    print(
        df[revenue_col].describe()
    )


if user_col is not None:

    print("\nUser statistics:")

    print(
        df[user_col].describe()
    )


# ================================================================
# 17. STATE ANALYSIS
# ================================================================

print("\n")
print("=" * 90)
print("STATE / REGION ANALYSIS")
print("=" * 90)


if state_col is not None:

    state_summary = df.groupby(
        state_col
    ).agg(

        Records=(
            state_col,
            "count"
        )
    )


    if revenue_col is not None:

        state_summary[
            "Total_Revenue"
        ] = df.groupby(
            state_col
        )[revenue_col].sum()


    if user_col is not None:

        state_summary[
            "Total_Users"
        ] = df.groupby(
            state_col
        )[user_col].sum()


    if demand_col is not None:

        state_summary[
            "Total_Demand"
        ] = df.groupby(
            state_col
        )[demand_col].sum()


    if revenue_col is not None and user_col is not None:

        state_summary[
            "Revenue_Per_User"
        ] = (
            state_summary["Total_Revenue"] /
            state_summary["Total_Users"].replace(
                0,
                np.nan
            )
        )


    print(
        state_summary.sort_values(
            "Total_Revenue"
            if "Total_Revenue" in state_summary.columns
            else "Records",
            ascending=False
        )
    )

else:

    state_summary = None

    print("State column not available.")


# ================================================================
# 18. CITY ANALYSIS
# ================================================================

print("\n")
print("=" * 90)
print("CITY ANALYSIS")
print("=" * 90)


if city_col is not None:

    city_summary = df.groupby(
        city_col
    ).agg(

        Records=(
            city_col,
            "count"
        )
    )


    if revenue_col is not None:

        city_summary[
            "Total_Revenue"
        ] = df.groupby(
            city_col
        )[revenue_col].sum()


    if user_col is not None:

        city_summary[
            "Total_Users"
        ] = df.groupby(
            city_col
        )[user_col].sum()


    if demand_col is not None:

        city_summary[
            "Total_Demand"
        ] = df.groupby(
            city_col
        )[demand_col].sum()


    if revenue_col is not None and user_col is not None:

        city_summary[
            "Revenue_Per_User"
        ] = (
            city_summary["Total_Revenue"] /
            city_summary["Total_Users"].replace(
                0,
                np.nan
            )
        )


    print(
        city_summary.sort_values(
            "Total_Revenue"
            if "Total_Revenue" in city_summary.columns
            else "Records",
            ascending=False
        ).head(20)
    )

else:

    city_summary = None

    print("City column not available.")


# ================================================================
# 19. POSTAL CODE ANALYSIS
# ================================================================

print("\n")
print("=" * 90)
print("POSTAL CODE ANALYSIS")
print("=" * 90)


if postal_col is not None:

    postal_summary = df.groupby(
        postal_col
    ).agg(

        Records=(
            postal_col,
            "count"
        )
    )


    if revenue_col is not None:

        postal_summary[
            "Total_Revenue"
        ] = df.groupby(
            postal_col
        )[revenue_col].sum()


    if user_col is not None:

        postal_summary[
            "Total_Users"
        ] = df.groupby(
            postal_col
        )[user_col].sum()


    print(
        postal_summary.head(20)
    )

else:

    postal_summary = None

    print(
        "Postal code column not available."
    )


# ================================================================
# 20. CREATE REGIONAL ANALYSIS
# ================================================================

print("\n")
print("=" * 90)
print("REGIONAL MARKET ANALYSIS")
print("=" * 90)


# Prefer city as the main geographic region.
# If city does not exist, use state.

if city_col is not None:

    region_col = city_col

elif state_col is not None:

    region_col = state_col

else:

    region_col = None


if region_col is not None:

    regional_summary = df.groupby(
        region_col
    ).agg(

        Records=(
            region_col,
            "count"
        )
    )


    if revenue_col is not None:

        regional_summary[
            "Revenue"
        ] = df.groupby(
            region_col
        )[revenue_col].sum()


    if user_col is not None:

        regional_summary[
            "Users"
        ] = df.groupby(
            region_col
        )[user_col].sum()


    if demand_col is not None:

        regional_summary[
            "Demand"
        ] = df.groupby(
            region_col
        )[demand_col].sum()


    # Average revenue

    if revenue_col is not None:

        regional_summary[
            "Average_Revenue"
        ] = df.groupby(
            region_col
        )[revenue_col].mean()


    # Revenue per user

    if revenue_col is not None and user_col is not None:

        regional_summary[
            "Revenue