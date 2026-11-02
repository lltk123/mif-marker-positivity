"""Read-only checks of published population, gate and artifact claims."""
import csv, hashlib, json, unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def rows(name):
 with (R/name).open(encoding="utf-8") as f:return list(csv.DictReader(f))
def load(name):return json.loads((R/name).read_text(encoding="utf-8"))
class PublishedResults(unittest.TestCase):
 def test_original_result_csv_content_preserved(self):
  selected=[x for x in rows("VERSION_INDEX.csv") if x["file"].startswith("results/") and x["file"].endswith(".csv")]
  self.assertEqual(len(selected),45)
  for x in selected:self.assertEqual(hashlib.sha256((R/x["file"]).read_bytes().replace(b"\r\n",b"\n")).hexdigest(),x["normalized_source_sha256"])
 def test_population_denominators_are_separate(self):
  full=[x for x in rows("results/ALL_EXTERNAL_SIX_METRICS.csv") if x["level"]=="pooled"];self.assertTrue(all(int(x["n"])==437480 for x in full))
  few=rows("results/ALL_FEWSHOT_POOLED.csv");self.assertTrue(all(int(x["n"])==433280 for x in few));self.assertEqual(437480-433280,42*100)
 def test_joint_gate1_not_rounded_to_pass(self):
  g=load("results/GOALS_AND_BOUNDARIES.json");sp=[x for x in g["Gate1"] if x["method"]=="spatial"][0];self.assertFalse(sp["passed_goal"]);self.assertLess(sp["AP"],.966);self.assertGreater(sp["F1"],.892);self.assertGreater(sp["weak_recall"],.48)
 def test_gate2_mean_and_individual_repeats_differ(self):
  x=[x for x in rows("results/ALL_FEWSHOT_POOLED.csv") if x["strategy"]=="active_boundary" and int(x["budget"])==20];self.assertEqual(len(x),3);self.assertEqual(sum(float(v["F1"])>=.902 and float(v["weak_recall"])>=.54 for v in x),2);self.assertGreater(sum(float(v["F1"]) for v in x)/3,.902);self.assertGreater(sum(float(v["weak_recall"]) for v in x)/3,.54)
 def test_internal_failure_stops_new_external_branch(self):
  gate=load("models/CONTINUATION_INTERNAL_GATE.json");self.assertFalse(gate["passed"]);self.assertLess(gate["ensemble_deltas"]["delta_F1"],0);self.assertGreater(gate["ensemble_deltas"]["delta_AP"],0);self.assertEqual(load("results/CONTINUATION_EXTERNAL_SKIPPED.json")["reason"],"development_gate_failed")
if __name__=="__main__":unittest.main(verbosity=2)
