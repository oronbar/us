import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_strain_pkls_by_prefix(patient_folder, xml_filename):
    prefix = os.path.splitext(os.path.basename(xml_filename))[0]
    endo_pkl = os.path.join(patient_folder, f"{prefix}_Strain-Endo_longitudinal_strain.pkl")
    epi_pkl = os.path.join(patient_folder, f"{prefix}_Strain-Epi_longitudinal_strain.pkl")
    dfs = {}
    if os.path.exists(endo_pkl):
        dfs['endo'] = pd.read_pickle(endo_pkl)
    else:
        print(f"File not found: {endo_pkl}")
    if os.path.exists(epi_pkl):
        dfs['epi'] = pd.read_pickle(epi_pkl)
    else:
        print(f"File not found: {epi_pkl}")
    return dfs, prefix

def plot_endo_epi_columns(df_endo, df_epi, n_cols=6, save_path=None, show_plot=True):
    if df_endo is None or df_epi is None:
        print("Both Endo and Epi DataFrames are required.")
        return
    # Use the last column as time
    time_endo = pd.to_numeric(df_endo[df_endo.columns[-1]], errors='coerce')
    time_epi = pd.to_numeric(df_epi[df_epi.columns[-1]], errors='coerce')
    # Use the same columns for both (up to n_cols, excluding time col)
    cols = df_endo.columns[:n_cols] if df_endo.shape[1] > n_cols else df_endo.columns[:-1]
    colors = plt.cm.get_cmap('tab10', len(cols))
    plt.figure(figsize=(12, 6))
    for idx, col in enumerate(cols):
        plt.plot(time_endo, pd.to_numeric(df_endo[col], errors='coerce'), color=colors(idx), linestyle='-', label=f'{col} (Endo)')
        plt.plot(time_epi, pd.to_numeric(df_epi[col], errors='coerce'), color=colors(idx), linestyle='--', label=f'{col} (Epi)')
    plt.xlabel(df_endo.columns[-1])
    plt.ylabel('Value')
    plt.title('Endo (solid) and Epi (dashed) - First {} Columns vs Time'.format(len(cols)))
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot Endo and Epi strain .pkl files for a patient using XML filename as prefix.")
    parser.add_argument("--xml_file", type=str, required=True, help="XML filename (or path) to use as prefix for .pkl files. The folder containing this file will be used as the patient folder.")
    parser.add_argument("--n_cols", type=int, help="Number of columns to plot (default 6)", default=6)
    parser.add_argument("--no_show", action="store_true", help="Suppress interactive plot display (useful for running as an exe or batch)")
    args = parser.parse_args()
    patient_folder = os.path.dirname(os.path.abspath(args.xml_file))
    dfs, prefix = load_strain_pkls_by_prefix(patient_folder, args.xml_file)
    df_endo = dfs.get('endo')
    df_epi = dfs.get('epi')
    # Compose the save path for the PNG
    png_filename = f"{prefix}_Endo_Epi_longitudinal_strain_plot.png"
    save_path = os.path.join(patient_folder, png_filename)
    plot_endo_epi_columns(df_endo, df_epi, n_cols=args.n_cols, save_path=save_path, show_plot=not args.no_show) 