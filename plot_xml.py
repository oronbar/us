import argparse
import subprocess
import sys
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process: parse XML and plot strain curves.")
    parser.add_argument("--xml_file", type=str, required=True, help="Path to the XML file to process.")
    parser.add_argument("--n_cols", type=int, default=6, help="Number of columns to plot (default 6)")
    parser.add_argument("--no_show", action="store_true", help="Suppress interactive plot display (useful for running as an exe or batch)")
    args = parser.parse_args()

    # Step 1: Run parse_xml.py
    print(f"Running parse_xml.py on {args.xml_file}...")
    result1 = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "parse_xml.py"), args.xml_file])
    if result1.returncode != 0:
        print("parse_xml.py failed.")
        sys.exit(result1.returncode)

    # Step 2: Run load_and_plot_strain.py
    print(f"Running load_and_plot_strain.py on {args.xml_file}...")
    plot_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "load_and_plot_strain.py"),
        "--xml_file", args.xml_file,
        "--n_cols", str(args.n_cols)
    ]
    if args.no_show:
        plot_cmd.append("--no_show")
    result2 = subprocess.run(plot_cmd)
    if result2.returncode != 0:
        print("load_and_plot_strain.py failed.")
        sys.exit(result2.returncode)

    print("Batch processing complete.") 