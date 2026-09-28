"""Check unchanged scientific source, R1 method source, and CSV numeric cells."""
import argparse,csv,hashlib,json,itertools
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','candidate','audit'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();old=a.source.resolve();new=a.candidate.resolve();audit=a.audit.resolve()
    checks={};numeric=0;csv_files=0
    core=['class_balancePM.cpp','class_balancePM.h','helpers.cpp','helpers.h','main.cpp','post.cpp','post.h','Makevars','Makevars.win']
    for name in core:
        assert (old/'code/BATTS/src'/name).read_bytes()==(new/'code/methods/ReviewPkg/src'/name).read_bytes(),name
    checks['r2_core_and_compiler_files_byte_identical']=len(core)
    r1=old/'reference/r1/code/methods/ReviewPkg';count=0
    for f in r1.rglob('*'):
        if f.is_file():
            assert f.read_bytes()==(new/f.relative_to(old)).read_bytes(),str(f);count+=1
    checks['r1_method_files_byte_identical']=count
    for f in old.rglob('*.csv'):
        rel=f.relative_to(old)
        if str(rel)=='SOURCE_MANIFEST.csv' or f.name=='output_checksums.csv':continue
        other=new/rel
        if not other.exists():continue
        csv_files+=1
        with f.open(newline='',encoding='utf-8-sig') as x,other.open(newline='',encoding='utf-8-sig') as y:
            for row1,row2 in itertools.zip_longest(csv.reader(x),csv.reader(y)):
                assert row1 is not None and row2 is not None and len(row1)==len(row2),str(rel)
                for v1,v2 in zip(row1,row2):
                    try:float(v1)
                    except ValueError:continue
                    assert v1==v2,(str(rel),v1,v2)
                    numeric+=1
    checks.update(csv_files_checked=csv_files,numeric_csv_cells_identical=numeric)
    mapping=json.loads((audit/'private_source_mapping.json').read_text(encoding='utf-8'))
    for row in mapping['files']:
        path=new/row['release_path'] if row['release_path'] else None
        row['final_release_sha256']=hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
        row['final_changed']=row['source_sha256']!=row['final_release_sha256']
    (audit/'private_final_source_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
    (audit/'content-validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
