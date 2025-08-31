import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Path to the folder containing the JSON files
folder_path_dataset = "../../dataset/"
folder_path_output = "./"

# Initialize counters and containers
total_files = 0
total_samples = 0
max_law_length = 0
min_law_length = float("inf")  # Initialize to infinity for min comparison
max_code_length = 0
min_code_length = float("inf")  # Initialize to infinity for min comparison
max_law_file = ""
min_law_file = ""
max_code_file = ""
min_code_file = ""

# DataFrame to store stats for each file
df_stats_file = pd.DataFrame(
    columns=[
        "Folder",
        "File",
        "Samples",
        "Max_Law_Length",
        "Min_Law_Length",
        "Max_Code_Length",
        "Min_Code_Length",
    ]
)

# DataFrame to store sample-level stats
df_stats_sample = pd.DataFrame(
    columns=["Folder", "File", "Sample_ID", "Law_Length", "Code_Length"]
)


# Function to analyze each JSON file
def analyze_file(file_path, foldername, filename):
    global total_samples, max_law_length, min_law_length, max_code_length, min_code_length
    global max_law_file, min_law_file, max_code_file, min_code_file

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load file as a JSON dictionary

        # Count the number of samples (assumes each key in the dictionary is a sample)
        samples = len(data)  # Each dictionary entry is considered a sample
        total_samples += samples

        # Initialize variables to track max/min lengths for the current file
        file_max_law_length = 0
        file_min_law_length = float("inf")
        file_max_code_length = 0
        file_min_code_length = float("inf")

        # Loop over each sample in the file
        for sample_id, sample in data.items():
            if sample_id == "metadata":
                continue
            law = sample.get("input", "")  # Extract the 'law' entry, if available
            code = sample.get("output", "")  # Extract the 'code' entry, if available

            # Calculate lengths
            law_length = len(law)
            code_length = len(code)

            # Update sample-level stats DataFrame
            global df_stats_sample
            df_stats_sample.loc[len(df_stats_sample)] = [
                foldername,
                filename,
                sample_id,
                law_length,
                code_length,
            ]

            # Check and update max/min length for law
            if law_length > max_law_length:
                max_law_length = law_length
                max_law_file = file_path
            if law_length < min_law_length:
                min_law_length = law_length
                min_law_file = file_path
            if law_length > file_max_law_length:
                file_max_law_length = law_length
            if law_length < file_min_law_length:
                file_min_law_length = law_length

            # Check and update max/min length for code
            if code_length > max_code_length:
                max_code_length = code_length
                max_code_file = file_path
            if code_length < min_code_length:
                min_code_length = code_length
                min_code_file = file_path
            if code_length > file_max_code_length:
                file_max_code_length = code_length
            if code_length < file_min_code_length:
                file_min_code_length = code_length

        # Append file-level stats for the current file to the DataFrame
        if file_min_law_length != float("inf") and file_min_code_length != float("inf"):
            df_stats_file.loc[len(df_stats_file)] = [
                foldername,
                filename,
                samples,
                file_max_law_length,
                file_min_law_length,
                file_max_code_length,
                file_min_code_length,
            ]


def save_dataframe_as_image(df, filename):
    """Save the DataFrame as an image with alternate row colors and adjusted column widths."""
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))  # Adjust size as needed
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    the_table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    # Set the font size
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Adjust column widths based on content
    for i in range(len(df.columns)):
        max_length = max(
            len(str(val)) for val in df[df.columns[i]].values
        )  # Get max length in the column
        the_table.auto_set_column_width(i)  # Automatically set the width
        the_table[i, 0].set_width(
            max_length * 0.1
        )  # Adjust width based on content (0.1 is a scaling factor)

    # Add alternate row colors
    for i, key in enumerate(the_table.get_celld().keys()):
        cell = the_table[key]
        if cell.get_text().get_text() != "":
            if key[0] == 0:  # Header row
                cell.set_facecolor("#b0b0b0")  # Darker gray for header
            elif key[0] % 2 == 1:  # Odd rows (1-based index for display)
                cell.set_facecolor("#d9d9d9")  # Light gray for odd rows
            else:  # Even rows
                cell.set_facecolor("#ffffff")  # White for even rows

    the_table.scale(1.2, 1.2)  # Scale the table

    plt.savefig(
        filename, bbox_inches="tight", dpi=300
    )  # Save the figure as a .png file
    plt.close(fig)


if __name__ == "__main__":

    if not os.path.exists(folder_path_output):
        os.makedirs(folder_path_output)

    # Loop through all files in the folder
    for foldername in os.listdir(folder_path_dataset):
        print(f"Analyzing folder: {foldername}")
        for filename in os.listdir(os.path.join(folder_path_dataset, foldername)):
            print(f"Analyzing filename: {filename}")
            if filename.endswith(".json"):
                total_files += 1
                analyze_file(
                    os.path.join(folder_path_dataset, foldername, filename),
                    foldername,
                    filename,
                )

    print("\nAnalysis complete!\n")

    # # Display the file-level statistics DataFrame
    # print("\nFile-wise statistics:")
    # display(df_stats_file)

    # Save the file-level statistics as an image
    save_dataframe_as_image(df_stats_file, folder_path_output + "file_statistics.png")

    # Optionally save the file-level statistics to a CSV file
    df_stats_file.to_csv(folder_path_output + "file_statistics.csv", index=False)

    # # Display the sample-level statistics DataFrame
    # print("\nSample-wise statistics:")
    # display(df_stats_sample)

    # Optionally save the sample-level statistics to a CSV file
    df_stats_sample.to_csv(folder_path_output + "sample_statistics.csv", index=False)

    # Describe the sample-level statistics, exporting a png image
    print("\nSample-wise statistics description:")
    df2 = df_stats_sample.describe().T
    display(df2)
