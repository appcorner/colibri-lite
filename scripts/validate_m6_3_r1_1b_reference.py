#!/usr/bin/env python3
"""Validate the immutable R1.1b code_newline checkpoint contract."""
import hashlib,json,sys
from pathlib import Path
def fail(message): raise ValueError(message)
def main():
    record=json.loads(Path(sys.argv[1]).read_text()); root=Path(sys.argv[2])
    if record['fixture']['id']!='code_newline' or record['fixture']['token_ids']!=[87,28,16,198]: fail('fixture mismatch')
    if record['revision']!='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39': fail('revision mismatch')
    if not record.get('existing_final_norm_and_fixed_logit_linkage'): fail('missing fixed-logit linkage')
    for checkpoint in record['checkpoints']:
        if checkpoint['name']!='layer0.selected_expert_ids' and not checkpoint.get('tolerance_ids'): fail('missing tolerance linkage')
    path=root/record['checkpoint_file']['path']; raw=path.read_bytes()
    if len(raw)!=record['checkpoint_file']['bytes'] or hashlib.sha256(raw).hexdigest()!=record['checkpoint_file']['sha256']: fail('checkpoint hash/length mismatch')
    if len(set(record['repeated_run_sha256']))!=1 or record['repeated_run_sha256'][0]!=record['checkpoint_file']['sha256']: fail('repeated-run mismatch')
    print(json.dumps({'status':'passed','reference_id':record['reference_id']}))
if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,OSError,IndexError) as error: print(f'reference validation error: {error}',file=sys.stderr);sys.exit(1)
