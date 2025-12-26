# Train Data Frame
import pandas as pd

try:
    df_train_csv = pd.read_csv(
        "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/Train_data.csv",
        names=["name", "label", "x_min", "y_min", "x_max", "y_max"],
    )

except FileNotFoundError:
    print("File not found. Please check the training csv file path.")

# sort the train dataframe by name
df_train_csv = df_train_csv.sort_values(by=["name"]).reset_index(drop=True)
print(df_train_csv.head())
print(df_train_csv.shape)


# Test Data Frame
try:
    df_test_csv = pd.read_csv(
        "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/Test_data.csv",
        names=["name", "label", "x_min", "y_min", "x_max", "y_max"],
    )
except FileNotFoundError:
    print("File not found. Please check the testing csv file path.")

# sort the test dataframe by name
df_test_csv = df_test_csv.sort_values(by=["name"]).reset_index(drop=True)
print(df_test_csv.head())
print(df_test_csv.shape)


def get_train_dataframe():
    """Returns the training dataframe
    that contains the bounding box coordinates and labels
    for each image in the training set."""
    return df_train_csv


def get_test_dataframe():
    """Returns the testing dataframe
    that contains the bounding box coordinates and labels
    for each image in the testing set."""
    return df_test_csv


print("Dataframes loaded successfully.")


def check_missing_rows(df, total_images):
    missing_sum = 0

    for i in range(680, 680 + total_images + 1):
        curName = str(i) + ".tif"
        if i not in df["name"].str.replace(".tif", "").astype(int).values:
            missing_sum += 1
            print(f"Missing row for image: {curName}")

    print(f"Total missing rows: {missing_sum}")


# Train Data Frame missing rows:
# Missing row for image: 63.tif
# Missing row for image: 111.tif
# Missing row for image: 114.tif
# Missing row for image: 124.tif
# Missing row for image: 159.tif
# Missing row for image: 166.tif
# Missing row for image: 338.tif
# Missing row for image: 344.tif
# Missing row for image: 366.tif
# Missing row for image: 367.tif
# Missing row for image: 379.tif
# Missing row for image: 382.tif
# Missing row for image: 390.tif
# Missing row for image: 395.tif
# Missing row for image: 517.tif
# Missing row for image: 530.tif
# Missing row for image: 556.tif
# Missing row for image: 613.tif
# Missing row for image: 642.tif
# Missing row for image: 680.tif

# Test Data Frame missing rows:
# Missing row for image: 688.tif
# Missing row for image: 731.tif
# Missing row for image: 749.tif
# Missing row for image: 756.tif
# Missing row for image: 765.tif
# Missing row for image: 777.tif
# Total missing rows: 6

# check_missing_rows(df_train_csv, 680)
# check_missing_rows(df_test_csv, 120)
