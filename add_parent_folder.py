import pandas as pd
import os

# Paths
excel_path = r"C:\Users\oronbarazani\OneDrive - Technion\DS\Cardio-Onco Echo SZMC\carasso_accs_Report2_with_parents.xlsx"
base_folder = r"D:\DS\Cardio-Onco Echo SZMC"
output_path = r"C:\Users\oronbarazani\OneDrive - Technion\DS\Cardio-Onco Echo SZMC\carasso_accs_Report2_with_parents_withdates.xlsx"

# Read Excel
df = pd.read_excel(excel_path)

# Get all parent folders
parent_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

match_count = 0

def find_parent_folder(study_uid):
    global match_count
    for parent in parent_folders:
        possible_path = os.path.join(base_folder, parent, str(study_uid))
        if os.path.exists(possible_path):
            match_count += 1
            return parent
    return None  # If not found

# Insert parent folder column before "Study Instance UID"
study_uid_idx = df.columns.get_loc("Study Instance UID")
df.insert(study_uid_idx, "Parent Folder", df["Study Instance UID"].apply(find_parent_folder))

# Remove entries where parent folder was not found
df = df[df["Parent Folder"].notna()]

print(f"Number of matches found: {match_count}")

# Save to new Excel file
df.to_excel(output_path, index=False)