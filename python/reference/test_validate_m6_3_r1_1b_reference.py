import json, subprocess, sys, tempfile, unittest
from pathlib import Path
class Tests(unittest.TestCase):
 def setUp(self): self.root=Path(__file__).resolve().parents[2]; self.record=self.root/'models/qwen3-30b-a3b/m6.3-r1-1b-code-newline-layer0-reference-v1.json'
 def invoke_validator(self,obj):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'r.json';p.write_text(json.dumps(obj));return subprocess.run([sys.executable,'scripts/validate_m6_3_r1_1b_reference.py',str(p),str(self.root)],cwd=self.root,capture_output=True,text=True)
 def test_valid(self): self.assertEqual(self.invoke_validator(json.loads(self.record.read_text())).returncode,0)
 def test_missing_linkage(self):
  x=json.loads(self.record.read_text());x['existing_final_norm_and_fixed_logit_linkage']='';self.assertNotEqual(self.invoke_validator(x).returncode,0)
 def test_revision(self):
  x=json.loads(self.record.read_text());x['revision']='x';self.assertNotEqual(self.invoke_validator(x).returncode,0)
if __name__=='__main__':unittest.main()
