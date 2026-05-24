import csv
import tempfile
import unittest
from pathlib import Path

from autostrain_anonymizer import collect_input_files, process_reports


SAMPLE_REPORT = """Export Date and Time,2026-03-17,13:09:38
CAP Name and Version,AutoStrainCap2,2.2.1.586242
Patient Name,"Doe, Jane"
Patient ID,12345
Patient Date of Birth,1970-01-02
Patient Sex,female
Study ID,98765
Study UID,1.2.3.4.5
Study Date and Time,2026-03-17,13:09:38

Results LvApical
==========================================================
GLS Endo Peak Avg,-18.55,%
"""


class AutoStrainAnonymizerTests(unittest.TestCase):
    def test_process_report_removes_patient_rows_and_writes_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "AutoStrainCap_test.txt"
            output_dir = root / "out"
            mapping_csv = output_dir / "mapping.csv"
            source.write_text(SAMPLE_REPORT, encoding="utf-8", newline="")

            results = process_reports([source], output_dir, mapping_csv, append_mapping=False)

            self.assertEqual(len(results), 1)
            output_text = (output_dir / source.name).read_text(encoding="utf-8")
            self.assertNotIn("Patient Name", output_text)
            self.assertNotIn("Patient ID", output_text)
            self.assertNotIn("Patient Date of Birth", output_text)
            self.assertNotIn("Patient Sex", output_text)
            self.assertIn("Study UID,1.2.3.4.5", output_text)

            with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertNotIn("processed_at", reader.fieldnames or [])
            self.assertIn("study_date_and_time", reader.fieldnames or [])
            self.assertIn("export_date_and_time", reader.fieldnames or [])
            self.assertEqual(rows[0]["study_uid"], "1.2.3.4.5")
            self.assertEqual(rows[0]["study_date_and_time"], "2026-03-17,13:09:38")
            self.assertEqual(rows[0]["export_date_and_time"], "2026-03-17,13:09:38")
            self.assertEqual(rows[0]["patient_name"], "Doe, Jane")
            self.assertEqual(rows[0]["patient_id"], "12345")
            self.assertEqual(rows[0]["patient_date_of_birth"], "1970-01-02")
            self.assertEqual(rows[0]["patient_sex"], "female")

    def test_process_report_reuses_mapping_csv_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "AutoStrainCap_test.txt"
            output_dir = root / "out"
            mapping_csv = output_dir / "mapping.csv"
            source.write_text(SAMPLE_REPORT, encoding="utf-8", newline="")

            process_reports([source], output_dir, mapping_csv, append_mapping=False)
            results = process_reports([source], output_dir, mapping_csv, append_mapping=True)

            self.assertEqual(results[0].skipped_reason, "already listed in mapping CSV")
            self.assertTrue((output_dir / source.name).exists())
            self.assertFalse((output_dir / "AutoStrainCap_test_2.txt").exists())

            with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_process_report_updates_legacy_mapping_columns_without_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "AutoStrainCap_test.txt"
            output_dir = root / "out"
            mapping_csv = output_dir / "mapping.csv"
            source.write_text(SAMPLE_REPORT, encoding="utf-8", newline="")
            output_dir.mkdir()
            existing_output = output_dir / source.name
            existing_output.write_text("already anonymized", encoding="utf-8")
            with mapping_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "processed_at",
                        "study_uid",
                        "study_id",
                        "patient_name",
                        "patient_id",
                        "patient_date_of_birth",
                        "patient_sex",
                        "source_file",
                        "anonymized_file",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "processed_at": "2026-05-12T21:07:20",
                        "study_uid": "1.2.3.4.5",
                        "study_id": "98765",
                        "patient_name": "Doe, Jane",
                        "patient_id": "12345",
                        "patient_date_of_birth": "1970-01-02",
                        "patient_sex": "female",
                        "source_file": str(source.resolve()),
                        "anonymized_file": str(existing_output.resolve()),
                    }
                )

            process_reports([source], output_dir, mapping_csv, append_mapping=True)

            self.assertEqual(existing_output.read_text(encoding="utf-8"), "already anonymized")
            self.assertFalse((output_dir / "AutoStrainCap_test_2.txt").exists())
            with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertNotIn("processed_at", reader.fieldnames or [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["study_date_and_time"], "2026-03-17,13:09:38")
            self.assertEqual(rows[0]["export_date_and_time"], "2026-03-17,13:09:38")

    def test_process_report_reuses_existing_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "AutoStrainCap_test.txt"
            output_dir = root / "out"
            mapping_csv = output_dir / "mapping.csv"
            source.write_text(SAMPLE_REPORT, encoding="utf-8", newline="")
            output_dir.mkdir()
            existing_output = output_dir / source.name
            existing_output.write_text("already anonymized", encoding="utf-8")

            results = process_reports([source], output_dir, mapping_csv, append_mapping=False)

            self.assertEqual(
                results[0].skipped_reason,
                "anonymized file already exists; reused existing file",
            )
            self.assertEqual(existing_output.read_text(encoding="utf-8"), "already anonymized")
            self.assertFalse((output_dir / "AutoStrainCap_test_2.txt").exists())

            with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["anonymized_file"], str(existing_output.resolve()))

    def test_collect_input_files_finds_txt_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            txt_file = nested / "report.txt"
            csv_file = nested / "report.csv"
            txt_file.write_text("x", encoding="utf-8")
            csv_file.write_text("x", encoding="utf-8")

            files = collect_input_files([root], recursive=True)

            self.assertEqual(files, [txt_file.resolve()])


if __name__ == "__main__":
    unittest.main()
