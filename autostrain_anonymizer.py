from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PATIENT_FIELDS = (
    "Patient Name",
    "Patient ID",
    "Patient Date of Birth",
    "Patient Sex",
)

REPORT_DETAIL_FIELDS = (
    "Study UID",
    "Study ID",
    "Study Date and Time",
    "Export Date and Time",
)

MAP_COLUMNS = (
    "study_uid",
    "study_id",
    "study_date_and_time",
    "export_date_and_time",
    "patient_name",
    "patient_id",
    "patient_date_of_birth",
    "patient_sex",
    "source_file",
    "anonymized_file",
)

TEXT_EXTENSIONS = {".txt", ".csv"}


@dataclass(frozen=True)
class ReportResult:
    source_file: Path
    anonymized_file: Path
    study_uid: str
    study_id: str
    study_date_and_time: str
    export_date_and_time: str
    patient_name: str
    patient_id: str
    patient_date_of_birth: str
    patient_sex: str
    removed_rows: int
    warnings: tuple[str, ...] = ()
    skipped_reason: str = ""


@dataclass(frozen=True)
class ParsedReport:
    anonymized_text: str
    encoding: str
    details: dict[str, str]
    removed_rows: int
    warnings: tuple[str, ...]


def detect_text_encoding(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return "utf-16"

    for encoding in ("utf-8", "cp1255", "cp1252"):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def read_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    encoding = detect_text_encoding(data)
    return data.decode(encoding), encoding


def parse_csv_row(line: str) -> list[str]:
    row_text = line.rstrip("\r\n")
    try:
        return next(csv.reader([row_text]))
    except csv.Error:
        return row_text.split(",")


def first_value(row: Sequence[str]) -> str:
    if len(row) < 2:
        return ""
    if len(row) == 2:
        return row[1].strip()
    return ",".join(cell.strip() for cell in row[1:])


def parse_report(source_file: Path) -> ParsedReport:
    text, encoding = read_text(source_file)
    kept_lines: list[str] = []
    details: dict[str, str] = {}
    removed_rows = 0

    for line in text.splitlines(keepends=True):
        row = parse_csv_row(line)
        key = row[0].strip().lstrip("\ufeff") if row else ""

        if key in PATIENT_FIELDS:
            details[key] = first_value(row)
            removed_rows += 1
            continue

        if key in REPORT_DETAIL_FIELDS:
            details[key] = first_value(row)

        kept_lines.append(line)

    warnings: list[str] = []
    if removed_rows != len(PATIENT_FIELDS):
        warnings.append(
            f"removed {removed_rows} patient rows; expected {len(PATIENT_FIELDS)}"
        )
    if not details.get("Study UID"):
        warnings.append("Study UID not found")

    return ParsedReport(
        anonymized_text="".join(kept_lines),
        encoding=encoding,
        details=details,
        removed_rows=removed_rows,
        warnings=tuple(warnings),
    )


def report_result_from_parsed(
    source_file: Path,
    output_file: Path,
    parsed: ParsedReport,
    skipped_reason: str = "",
) -> ReportResult:
    details = parsed.details
    return ReportResult(
        source_file=source_file,
        anonymized_file=output_file,
        study_uid=details.get("Study UID", ""),
        study_id=details.get("Study ID", ""),
        study_date_and_time=details.get("Study Date and Time", ""),
        export_date_and_time=details.get("Export Date and Time", ""),
        patient_name=details.get("Patient Name", ""),
        patient_id=details.get("Patient ID", ""),
        patient_date_of_birth=details.get("Patient Date of Birth", ""),
        patient_sex=details.get("Patient Sex", ""),
        removed_rows=parsed.removed_rows,
        warnings=parsed.warnings,
        skipped_reason=skipped_reason,
    )


def anonymize_report(source_file: Path, output_file: Path) -> ReportResult:
    parsed = parse_report(source_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(parsed.anonymized_text, encoding=parsed.encoding, newline="")

    return report_result_from_parsed(source_file, output_file, parsed)


def mapping_row(result: ReportResult) -> dict[str, str]:
    return {
        "study_uid": result.study_uid,
        "study_id": result.study_id,
        "study_date_and_time": result.study_date_and_time,
        "export_date_and_time": result.export_date_and_time,
        "patient_name": result.patient_name,
        "patient_id": result.patient_id,
        "patient_date_of_birth": result.patient_date_of_birth,
        "patient_sex": result.patient_sex,
        "source_file": str(result.source_file),
        "anonymized_file": str(result.anonymized_file),
    }


def read_mapping_csv(mapping_csv: Path) -> list[dict[str, str]]:
    if not mapping_csv.exists() or mapping_csv.stat().st_size == 0:
        return []

    with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def csv_path_value(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def path_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def mapping_identity_keys(row: dict[str, str], base_dir: Path) -> set[str]:
    keys: set[str] = set()
    for column in ("source_file", "anonymized_file"):
        value = (row.get(column) or "").strip()
        if not value:
            continue
        mapped_path = csv_path_value(value, base_dir)
        keys.add(f"path:{path_key(mapped_path)}")
        keys.add(f"name:{mapped_path.name.lower()}")
    return keys


def find_existing_mapping_row(
    source_file: Path, rows: Sequence[dict[str, str]], base_dir: Path
) -> dict[str, str] | None:
    source_keys = {f"path:{path_key(source_file)}", f"name:{source_file.name.lower()}"}
    for row in rows:
        if source_keys & mapping_identity_keys(row, base_dir):
            return row
    return None


def result_from_mapping_row(
    source_file: Path, row: dict[str, str], output_dir: Path, base_dir: Path
) -> ReportResult:
    anonymized_value = (row.get("anonymized_file") or "").strip()
    anonymized_file = (
        csv_path_value(anonymized_value, base_dir)
        if anonymized_value
        else (output_dir / source_file.name).resolve()
    )
    return ReportResult(
        source_file=source_file,
        anonymized_file=anonymized_file,
        study_uid=row.get("study_uid", ""),
        study_id=row.get("study_id", ""),
        study_date_and_time=row.get("study_date_and_time", ""),
        export_date_and_time=row.get("export_date_and_time", ""),
        patient_name=row.get("patient_name", ""),
        patient_id=row.get("patient_id", ""),
        patient_date_of_birth=row.get("patient_date_of_birth", ""),
        patient_sex=row.get("patient_sex", ""),
        removed_rows=0,
        skipped_reason="already listed in mapping CSV",
    )


def refresh_mapping_row_from_source(
    row: dict[str, str], source_file: Path, output_dir: Path, base_dir: Path
) -> None:
    existing_result = result_from_mapping_row(source_file, row, output_dir, base_dir)
    parsed_result = report_result_from_parsed(
        source_file,
        existing_result.anonymized_file,
        parse_report(source_file),
        skipped_reason=existing_result.skipped_reason,
    )
    parsed_row = mapping_row(parsed_result)
    for column in MAP_COLUMNS:
        if column in ("source_file", "anonymized_file"):
            continue
        value = parsed_row.get(column, "")
        if value:
            row[column] = value


def existing_output_files(output_dir: Path) -> dict[str, Path]:
    if not output_dir.exists():
        return {}
    return {
        child.name.lower(): child.resolve()
        for child in output_dir.iterdir()
        if child.is_file()
    }


def write_mapping_csv(
    mapping_csv: Path,
    results: Sequence[ReportResult],
    append: bool = True,
    existing_rows: Sequence[dict[str, str]] | None = None,
) -> None:
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    if append:
        source_rows = (
            list(existing_rows)
            if existing_rows is not None
            else read_mapping_csv(mapping_csv)
        )
        for row in source_rows:
            normalized = {column: row.get(column, "") for column in MAP_COLUMNS}
            rows.append(normalized)
            seen.update(mapping_identity_keys(normalized, mapping_csv.parent))

    for result in results:
        row = mapping_row(result)
        keys = mapping_identity_keys(row, mapping_csv.parent)
        if keys & seen:
            continue
        rows.append(row)
        seen.update(keys)

    with mapping_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def collect_input_files(
    inputs: Iterable[Path], recursive: bool = True, include_csv: bool = False
) -> list[Path]:
    files: list[Path] = []
    allowed_exts = TEXT_EXTENSIONS if include_csv else {".txt"}

    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            files.extend(
                child
                for child in iterator
                if child.is_file() and child.suffix.lower() in allowed_exts
            )
        else:
            raise FileNotFoundError(f"Input does not exist: {input_path}")

    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files, key=lambda file_path: str(file_path).lower())


def process_reports(
    files: Sequence[Path],
    output_dir: Path,
    mapping_csv: Path,
    append_mapping: bool = True,
) -> list[ReportResult]:
    if not files:
        raise ValueError("No input reports selected.")

    output_dir = output_dir.expanduser().resolve()
    mapping_csv = mapping_csv.expanduser().resolve()
    results: list[ReportResult] = []
    new_results: list[ReportResult] = []
    existing_rows = read_mapping_csv(mapping_csv) if append_mapping else []
    outputs_by_name = existing_output_files(output_dir)

    for source_file in files:
        source = source_file.expanduser().resolve()
        mapped_row = find_existing_mapping_row(source, existing_rows, mapping_csv.parent)
        if mapped_row is not None:
            refresh_mapping_row_from_source(
                mapped_row, source, output_dir, mapping_csv.parent
            )
            results.append(
                result_from_mapping_row(source, mapped_row, output_dir, mapping_csv.parent)
            )
            continue

        existing_output = outputs_by_name.get(source.name.lower())
        if existing_output is not None:
            parsed = parse_report(source)
            result = report_result_from_parsed(
                source,
                existing_output,
                parsed,
                skipped_reason="anonymized file already exists; reused existing file",
            )
            results.append(result)
            new_results.append(result)
            continue

        output_file = output_dir / source.name
        result = anonymize_report(source, output_file)
        results.append(result)
        new_results.append(result)
        outputs_by_name[source.name.lower()] = output_file.resolve()

    write_mapping_csv(
        mapping_csv,
        new_results,
        append=append_mapping,
        existing_rows=existing_rows,
    )
    return results


def open_in_file_manager(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        pass


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("AutoStrainCap Anonymizer")
    root.geometry("820x560")
    root.minsize(740, 500)

    selected_files: list[Path] = []

    output_dir_var = tk.StringVar(value=str((Path.cwd() / "anonymized_reports").resolve()))
    mapping_csv_var = tk.StringVar(
        value=str((Path.cwd() / "anonymized_reports" / "anonymization_mapping.csv").resolve())
    )
    recursive_var = tk.BooleanVar(value=True)
    append_mapping_var = tk.BooleanVar(value=True)

    def update_listbox() -> None:
        file_list.delete(0, tk.END)
        for file_path in selected_files:
            file_list.insert(tk.END, str(file_path))
        count_var.set(f"{len(selected_files)} report file(s) selected")

    def add_paths(paths: Iterable[Path]) -> None:
        known = {path.resolve() for path in selected_files}
        for path in paths:
            resolved = path.resolve()
            if resolved not in known:
                selected_files.append(resolved)
                known.add(resolved)

        if selected_files:
            first_parent = selected_files[0].parent
            if output_dir_var.get() == str((Path.cwd() / "anonymized_reports").resolve()):
                out_dir = first_parent / "anonymized_reports"
                output_dir_var.set(str(out_dir))
                mapping_csv_var.set(str(out_dir / "anonymization_mapping.csv"))

        update_listbox()

    def add_files() -> None:
        paths = filedialog.askopenfilenames(
            title="Select AutoStrainCap text reports",
            filetypes=(("Text reports", "*.txt"), ("All files", "*.*")),
        )
        add_paths(Path(path) for path in paths)

    def add_folder() -> None:
        folder = filedialog.askdirectory(title="Select folder with AutoStrainCap reports")
        if not folder:
            return
        try:
            files = collect_input_files([Path(folder)], recursive=recursive_var.get())
        except OSError as exc:
            messagebox.showerror("Folder error", str(exc))
            return
        if not files:
            messagebox.showinfo("No reports found", "No .txt reports were found in that folder.")
            return
        add_paths(files)

    def remove_selected() -> None:
        indexes = list(file_list.curselection())
        for index in reversed(indexes):
            selected_files.pop(index)
        update_listbox()

    def clear_files() -> None:
        selected_files.clear()
        update_listbox()

    def browse_output_dir() -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        output_dir_var.set(folder)
        if not mapping_csv_var.get():
            mapping_csv_var.set(str(Path(folder) / "anonymization_mapping.csv"))

    def browse_mapping_csv() -> None:
        path = filedialog.asksaveasfilename(
            title="Select mapping CSV",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="anonymization_mapping.csv",
        )
        if path:
            mapping_csv_var.set(path)

    def log(message: str) -> None:
        log_box.configure(state="normal")
        log_box.insert(tk.END, message + "\n")
        log_box.see(tk.END)
        log_box.configure(state="disabled")

    def process_selected() -> None:
        if not selected_files:
            messagebox.showwarning("Missing input", "Add at least one report file or folder.")
            return
        if not output_dir_var.get():
            messagebox.showwarning("Missing output", "Choose an output folder.")
            return
        if not mapping_csv_var.get():
            messagebox.showwarning("Missing mapping CSV", "Choose a mapping CSV path.")
            return

        process_button.configure(state="disabled")
        root.update_idletasks()
        try:
            results = process_reports(
                selected_files,
                Path(output_dir_var.get()),
                Path(mapping_csv_var.get()),
                append_mapping=append_mapping_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Anonymization failed", str(exc))
            log(f"ERROR: {exc}")
            return
        finally:
            process_button.configure(state="normal")

        warning_count = sum(1 for result in results if result.warnings)
        log(f"Processed {len(results)} report(s). Mapping CSV: {mapping_csv_var.get()}")
        for result in results:
            status_parts = [result.skipped_reason] if result.skipped_reason else []
            status_parts.extend(result.warnings)
            status = "; ".join(status_parts) if status_parts else "OK"
            log(f"{result.source_file.name} -> {result.anonymized_file.name} [{status}]")
        messagebox.showinfo(
            "Done",
            f"Processed {len(results)} report(s).\nWarnings: {warning_count}\n\nMapping CSV:\n{mapping_csv_var.get()}",
        )

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    top_frame = ttk.Frame(root, padding=(12, 12, 12, 6))
    top_frame.grid(row=0, column=0, sticky="ew")
    top_frame.columnconfigure(4, weight=1)

    ttk.Button(top_frame, text="Add Files", command=add_files).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(top_frame, text="Add Folder", command=add_folder).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(top_frame, text="Remove Selected", command=remove_selected).grid(row=0, column=2, padx=(0, 8))
    ttk.Button(top_frame, text="Clear", command=clear_files).grid(row=0, column=3, padx=(0, 12))
    count_var = tk.StringVar(value="0 report file(s) selected")
    ttk.Label(top_frame, textvariable=count_var).grid(row=0, column=4, sticky="w")

    list_frame = ttk.Frame(root, padding=(12, 0, 12, 8))
    list_frame.grid(row=1, column=0, sticky="nsew")
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)

    file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    file_list.grid(row=0, column=0, sticky="nsew")
    file_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=file_list.yview)
    file_scroll.grid(row=0, column=1, sticky="ns")
    file_list.configure(yscrollcommand=file_scroll.set)

    options_frame = ttk.Frame(root, padding=(12, 0, 12, 8))
    options_frame.grid(row=2, column=0, sticky="ew")
    options_frame.columnconfigure(1, weight=1)

    ttk.Label(options_frame, text="Output folder").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(options_frame, textvariable=output_dir_var).grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=4)
    ttk.Button(options_frame, text="Browse", command=browse_output_dir).grid(row=0, column=2, pady=4)

    ttk.Label(options_frame, text="Mapping CSV").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(options_frame, textvariable=mapping_csv_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=4)
    ttk.Button(options_frame, text="Browse", command=browse_mapping_csv).grid(row=1, column=2, pady=4)

    ttk.Checkbutton(
        options_frame,
        text="Search folders recursively",
        variable=recursive_var,
    ).grid(row=2, column=0, sticky="w", pady=(4, 0))
    ttk.Checkbutton(
        options_frame,
        text="Append to existing mapping CSV",
        variable=append_mapping_var,
    ).grid(row=2, column=1, sticky="w", pady=(4, 0))

    bottom_frame = ttk.Frame(root, padding=(12, 0, 12, 12))
    bottom_frame.grid(row=3, column=0, sticky="ew")
    bottom_frame.columnconfigure(0, weight=1)

    log_box = tk.Text(bottom_frame, height=7, state="disabled", wrap="word")
    log_box.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

    process_button = ttk.Button(bottom_frame, text="Anonymize Reports", command=process_selected)
    process_button.grid(row=1, column=1, sticky="e", padx=(8, 0))
    ttk.Button(
        bottom_frame,
        text="Open Output Folder",
        command=lambda: open_in_file_manager(Path(output_dir_var.get())),
    ).grid(row=1, column=2, sticky="e", padx=(8, 0))

    update_listbox()
    root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove patient-identifying rows from AutoStrainCap text reports."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Report files or folders. If omitted, the GUI opens.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("anonymized_reports"),
        help="Folder for anonymized reports.",
    )
    parser.add_argument(
        "-m",
        "--mapping-csv",
        type=Path,
        default=None,
        help="CSV file that receives study UID to patient-detail mapping.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search input folders recursively.",
    )
    parser.add_argument(
        "--overwrite-mapping",
        action="store_true",
        help="Overwrite the mapping CSV instead of appending rows.",
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Also include .csv files when scanning folders.",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.inputs:
        run_gui()
        return 0

    output_dir = args.output_dir
    mapping_csv = args.mapping_csv or (output_dir / "anonymization_mapping.csv")
    files = collect_input_files(
        args.inputs,
        recursive=not args.no_recursive,
        include_csv=args.include_csv,
    )
    results = process_reports(
        files,
        output_dir,
        mapping_csv,
        append_mapping=not args.overwrite_mapping,
    )

    for result in results:
        status_parts = [result.skipped_reason] if result.skipped_reason else []
        status_parts.extend(result.warnings)
        status_text = f" ({'; '.join(status_parts)})" if status_parts else ""
        print(f"{result.source_file} -> {result.anonymized_file}{status_text}")
    print(f"Mapping CSV: {mapping_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
