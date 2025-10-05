import pandas as pd

# Path to the Excel file
excel_path = r"C:\Users\oronbarazani\OneDrive - Technion\DS\Cardio-Onco Echo SZMC\carasso_accs_Report2_with_parents.xlsx"
output_path = r"C:\Users\oronbarazani\OneDrive - Technion\DS\Cardio-Onco Echo SZMC\carasso_accs_Report2_with_parents_with_time.xlsx"

# Read Excel file
df = pd.read_excel(excel_path)

# Format date columns
def format_date(date):
    try:
        dt = pd.to_datetime(date)
        return dt.strftime("%Y_%m_%d_%H_%M_%S")
    except Exception:
        return ""

df["Study Date"] = df["Study Date"].apply(format_date)
df["Birth Date"] = df["Birth Date"].apply(format_date)

# Save to new Excel file
df.to_excel(output_path, index=False)