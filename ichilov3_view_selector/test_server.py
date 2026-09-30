"""Focused tests for visit grouping, selection constraints, and revision backups."""

import json
import tempfile
import unittest
from pathlib import Path

from server import ReviewData, cine_fps, file_id
from suggestions import MODEL_VIEWS, choose_distinct


class ReviewDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ichilov_view_selector_test_")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
        audit = root / "audit"
        audit.mkdir()
        self.output = root / "decisions"
        (audit / "visits.json").write_text(json.dumps([{
            "patient": "TEST-1", "date": "2020-02-03", "strain_report": "Yes",
            "matching_dicom_folder": "Yes", "tomtec_bookmark": "No",
            "resolved_views": 0, "notes": "", "dicom_folders": "E:\\test",
            "session_selection": "",
        }]), encoding="utf-8")
        self.paths = [rf"E:\test\wrong_folder\DCM000{i}.dcm" for i in range(3)]
        with (audit / "inventory.jsonl").open("w", encoding="utf-8") as handle:
            for i, path in enumerate(self.paths):
                handle.write(json.dumps({
                    "status": "ok", "path": path, "patient_folder": "TEST-1",
                    "date_folder": "2020_02_04", "StudyDate": "20200203",
                    "NumberOfFrames": "40", "InstanceNumber": str(i+1),
                    "SOPInstanceUID": f"1.2.3.{i}", "StudyInstanceUID": "1.2.3",
                }) + "\n")
        self.data = ReviewData(audit, self.output)

    def test_study_date_and_unique_selection(self):
        key = "TEST-1|2020-02-03"
        self.assertEqual(len(self.data.visit_detail(key)["files"]), 3)
        ids = [file_id(path) for path in self.paths]
        payload = {"visit_key": key, "views": dict(zip(("A2C", "A3C", "A4C"), ids)),
                   "reviewer": "Tester", "note": "", "unable": False}
        self.assertEqual(self.data.save(payload)["status"], "complete")
        self.assertEqual(self.data.export_rows()[0]["A2C_SOPInstanceUID"], "1.2.3.0")
        payload["views"]["A4C"] = ids[0]
        with self.assertRaisesRegex(ValueError, "different DICOM"):
            self.data.save(payload)

    def test_revisions_are_backed_up(self):
        payload = {"visit_key": "TEST-1|2020-02-03", "views": {v: "" for v in ("A2C", "A3C", "A4C")},
                   "reviewer": "Tester", "note": "checking", "unable": False}
        self.data.save(payload)
        payload["note"] = "revised"
        self.data.save(payload)
        backups = list(self.output.glob("decisions.backup.*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8"))["decisions"][payload["visit_key"]]["note"], "checking")
        self.assertEqual(self.data.decisions[payload["visit_key"]]["note"], "revised")

    def test_cine_uses_dicom_timing(self):
        self.assertAlmostEqual(cine_fps({"FrameTime": 20}), 50)
        self.assertEqual(cine_fps({"CineRate": 25}), 25)
        self.assertEqual(cine_fps({}), 30)

    def test_suggestion_abstains_from_weak_view(self):
        def row(identity, view, probability, agreement=1):
            probs = [0.0] * len(MODEL_VIEWS)
            probs[MODEL_VIEWS.index(view)] = probability
            return {"id": identity, "status": "ok", "bmode_candidate": True,
                    "predicted_view": view, "probabilities": probs,
                    "frame_agreement": {v: agreement if v == view else 0 for v in ("A2C", "A3C", "A4C")},
                    "quality": {"quality_proxy": .6}}

        selected, alternatives = choose_distinct([
            row("two", "A2C", .9), row("three", "A3C", .3), row("four", "A4C", .8)])
        self.assertEqual(selected, {"A2C": "two", "A3C": "", "A4C": "four"})
        self.assertEqual(alternatives["A3C"][0]["id"], "three")

    def test_approved_suggestion_provenance(self):
        payload = {"visit_key": "TEST-1|2020-02-03",
                   "views": {"A2C": file_id(self.paths[0]), "A3C": "", "A4C": ""},
                   "selection_source": {"A2C": "suggested", "A3C": "", "A4C": ""},
                   "reviewer": "Tester", "note": "", "unable": False}
        self.assertEqual(self.data.save(payload)["selection_source"]["A2C"], "suggested")
        self.assertEqual(self.data.export_rows()[0]["A2C_selection_source"], "suggested")

    def test_invalid_visit_clears_views_and_can_be_revised(self):
        key = "TEST-1|2020-02-03"
        ids = [file_id(path) for path in self.paths]
        complete = {"visit_key": key, "views": dict(zip(("A2C", "A3C", "A4C"), ids)),
                    "reviewer": "Tester", "note": "", "invalid": False}
        self.data.save(complete)
        invalid = {"visit_key": key, "views": {view: "" for view in ("A2C", "A3C", "A4C")},
                   "reviewer": "Tester", "note": "", "invalid": True}
        decision = self.data.save(invalid)
        self.assertEqual(decision["status"], "invalid")
        self.assertTrue(decision["note"])
        self.assertEqual(set(decision["views"].values()), {""})
        self.assertEqual(self.data.export_rows()[0]["status"], "invalid")
        self.assertEqual(self.data.save(complete)["status"], "complete")
        invalid["views"]["A2C"] = ids[0]
        with self.assertRaisesRegex(ValueError, "Clear the selections"):
            self.data.save(invalid)

    def test_tomtec_choice_must_match_bookmark_view(self):
        key = "TEST-1|2020-02-03"
        self.data.visits[key]["tomtec_bookmark"] = "Yes"
        self.data.visits[key]["audit_views"]["A2C"] = self.paths[0]
        payload = {"visit_key": key, "views": {"A2C": file_id(self.paths[0]), "A3C": "", "A4C": ""},
                   "selection_source": {"A2C": "tomtec"}, "reviewer": "Tester", "note": ""}
        self.assertEqual(self.data.save(payload)["selection_source"]["A2C"], "tomtec")
        payload["views"]["A2C"] = file_id(self.paths[1])
        with self.assertRaisesRegex(ValueError, "TOMTEC provenance"):
            self.data.save(payload)


if __name__ == "__main__":
    unittest.main()
