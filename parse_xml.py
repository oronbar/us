import xml.etree.ElementTree as ET
import pandas as pd
import os
import argparse

def extract_longitudinal_strain(xml_path, sheet_names):
    ns = {
        'ss': 'urn:schemas-microsoft-com:office:spreadsheet'
    }
    tree = ET.parse(xml_path)
    root = tree.getroot()
    dfs = {}

    xml_base = os.path.splitext(os.path.basename(xml_path))[0]

    for sheet_name in sheet_names:
        worksheet = root.find(f".//ss:Worksheet[@ss:Name='{sheet_name}']", ns)
        if worksheet is None:
            print(f"Sheet {sheet_name} not found.")
            continue
        table = worksheet.find(".//ss:Table", ns)
        rows = table.findall(".//ss:Row", ns)
        # Find the row index for "Longitudinal Strain"
        start_idx = None
        for i, row in enumerate(rows):
            cells = row.findall(".//ss:Cell/ss:Data", ns)
            if cells and "Longitudinal Strain" in (cells[0].text or ""):
                start_idx = i
                break
        if start_idx is None:
            print(f"'Longitudinal Strain' not found in {sheet_name}")
            continue
        # Collect all rows after the marker until the first empty row
        data = []
        for row in rows[start_idx + 1:]:
            values = [cell.text for cell in row.findall(".//ss:Cell/ss:Data", ns)]
            if not any(values):  # Stop at first empty row
                break
            data.append(values)
        # Transpose so that first column is header, rest are data columns
        if not data:
            print(f"No data found after 'Longitudinal Strain' in {sheet_name}")
            continue
        # The first column of each row is the header, the rest are the data for that header
        headers = [row[0] for row in data]
        # Find the maximum number of data points in any row (for padding)
        max_len = max(len(row) for row in data)
        # Pad rows to the same length
        padded_data = [row[1:] + [None]*(max_len-1-len(row[1:])) for row in data]
        # Now, columns = headers, rows = time/sample points
        df = pd.DataFrame(padded_data).T
        df.columns = headers
        df = df.reset_index(drop=True)
        dfs[sheet_name.strip()] = df
        # Save DataFrame to .pkl file with xml base name as prefix
        base_dir = os.path.dirname(xml_path)
        pkl_filename = os.path.join(base_dir, f"{xml_base}_{sheet_name.strip().replace(' ', '_')}_longitudinal_strain.pkl")
        df.to_pickle(pkl_filename)
        print(f"Saved {sheet_name.strip()} DataFrame to {pkl_filename}")
    return dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract longitudinal strain data from Excel 2003 XML file.")
    parser.add_argument("xml_file", type=str, help="Path to the XML file")
    args = parser.parse_args()
    sheet_names = [" Strain-Endo", " Strain-Myo", " Strain-Epi"]
    dfs = extract_longitudinal_strain(args.xml_file, sheet_names)
    for name, df in dfs.items():
        print(f"Sheet: {name}")
        print(df)

# To load a saved DataFrame later:
# df = pd.read_pickle('foo_Strain-Endo_longitudinal_strain.pkl')
