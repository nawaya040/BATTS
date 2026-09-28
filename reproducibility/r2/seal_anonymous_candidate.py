"""Seal an anonymous candidate and optionally create its verified ZIP.

Private source mapping stays outside the release. Refuse any unrecognized
binary payload or direct author/project identifier in release text.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

BLOCK = re.compile(r'BATTS|nawaya|Awaya|arxiv\.org/abs/2508\.03059|[A-Za-z]:[/\\]Users[/\\]|/Users/|/home/',re.I)
BINARY = {'.rds','.png','.pdf'}
def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('--zip',type=Path);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True,help='Immutable input candidate for adapting RDS hash references')
    a=p.parse_args();root=a.candidate.resolve(strict=True)
    source=a.source.resolve(strict=True)
    hash_map={}
    for old in (source/'results/r2/figure_details/boosting').glob('*.rds'):
        new=root/old.relative_to(source)
        hash_map[sha(old.read_bytes())]=sha(new.read_bytes())
    for f in (root/'results/r2/figure_details/metadata').glob('*.txt'):
        s=f.read_text(encoding='utf-8');t=s
        for old,new in hash_map.items():t=t.replace(old,new)
        if t!=s:f.write_text(t,encoding='utf-8')
    checksum=root/'results/r2/figure_details/boosting/output_checksums.csv'
    with checksum.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        path=checksum.parent/(row['job_id']+'.rds')
        if path.exists():row['output_sha256']=sha(path.read_bytes());row['output_bytes']=str(path.stat().st_size)
    with checksum.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Plot scripts pin input metadata as well as numerical data. Refresh those
    # input hashes after identity-only edits, retaining the integrity assertions.
    input_hash_map = {}
    for old in source.rglob('*'):
        if not old.is_file(): continue
        rel = old.relative_to(source)
        if rel.parts[0] not in {'results', 'reference'}: continue
        new = root/rel
        if new.is_file():
            before, after = sha(old.read_bytes()), sha(new.read_bytes())
            if before != after: input_hash_map[before] = after
    for f in (root/'code').rglob('*'):
        if not f.is_file() or f.suffix.lower() not in {'.r','.py','.ps1'}: continue
        s=f.read_text(encoding='utf-8');t=s
        for old,new in input_hash_map.items(): t=t.replace(old,new)
        if t!=s: f.write_text(t,encoding='utf-8')
    records=[]
    for f in sorted(root.rglob('*')):
        if not f.is_file() or f.name=='SOURCE_MANIFEST.csv':continue
        rel=f.relative_to(root).as_posix();data=f.read_bytes()
        if BLOCK.search(rel):raise SystemExit('Identifier in file name: '+rel)
        if f.suffix.lower() not in BINARY:
            text=data.decode('utf-8-sig')
            if BLOCK.search(text):raise SystemExit('Identifier in text: '+rel)
        records.append(dict(release_path=rel,release_bytes=len(data),release_sha256=sha(data)))
    manifest=root/'SOURCE_MANIFEST.csv'
    with manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['release_path','release_bytes','release_sha256']);w.writeheader();w.writerows(records)
    result=dict(files=len(records),manifest_sha256=sha(manifest.read_bytes()),text_identifier_scan='PASS')
    if a.zip:
        if a.zip.exists():raise SystemExit('Refusing to overwrite ZIP')
        expected={v['release_path']:v['release_sha256'] for v in records};expected['SOURCE_MANIFEST.csv']=sha(manifest.read_bytes())
        with ZipFile(a.zip,'x',ZIP_DEFLATED,compresslevel=6) as z:
            for rel in sorted(expected):z.write(root/rel,root.name+'/'+rel)
        with ZipFile(a.zip) as z:
            assert set(z.namelist())=={root.name+'/'+k for k in expected}
            for rel,h in expected.items():assert sha(z.read(root.name+'/'+rel))==h,rel
        result.update(zip_sha256=sha(a.zip.read_bytes()),zip_bytes=a.zip.stat().st_size,zip_members=len(expected),zip_root=root.name)
    a.audit.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
