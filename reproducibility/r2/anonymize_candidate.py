"""Build an anonymous R2 copy; retain identifying provenance outside the release.

The input candidate is immutable. RDS metadata is adapted by the paired R tool.
Run seal_anonymous_candidate.py after all validation records are finalized.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = {'.r', '.rmd', '.md', '.txt', '.csv', '.py', '.ps1', '.cpp', '.h', '.json', '.rd', '.tex', '.dput', '.win', ''}
PACKAGE_SHA = '6f625bad83702b36e5480be1ed1343258a9b075a'
WORKFLOW_SHA = 'f921525711faef2a326c561820749e0c70f1bf42'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def opaque(value):
    return hashlib.sha1(('anonymous-r2-source:' + value).encode()).hexdigest()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one occurrence: {old[:100]!r}')
    return text.replace(old, new)

def neutral(text):
    text = text.replace('code/BATTS/', 'code/methods/ReviewPkg/')
    text = text.replace('batts-lib', 'reviewpkg-lib')
    text = re.sub(r'\bboots\b', 'fit_boosting_model', text)
    text = re.sub(r'\bbatts\b', 'fit_bayesian_model', text)
    text = re.sub(r'\beval_balance_weight\b', 'evaluate_density_ratio', text)
    text = text.replace('BATTS', 'ReviewPkg').replace('batts', 'reviewpkg')
    text = text.replace('balancePM authors', 'Anonymous authors')
    text = re.sub(r'(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])', lambda m: opaque(m[0]), text)
    return text

def adapt_workflow(rel, text):
    text = text.replace('code-commit', 'release-id').replace('code_commit', 'release_id')
    text = text.replace('commit_hash', 'source_revision_id')
    text = text.replace('git_commit', 'release_id').replace('require_clean_git', 'require_release')
    text = text.replace('RemoteSha', 'Review-Source-ID')
    text = text.replace('Git commit', 'anonymous release ID').replace('Git worktree', 'release manifest')
    text = text.replace('report commit', 'report Review-Source-ID')
    if rel.endswith('/coverage_common.R') or rel.endswith('/boosting_selection_common.R'):
        prefix = 'coverage' if rel.endswith('/coverage_common.R') else 'boosting'
        start = text.index(prefix + '_release_id <- function(')
        end = text.index(prefix + ('_package_versions' if prefix == 'coverage' else '_load_packages'), start)
        text = text[:start] + f'''{prefix}_release_id <- function(project_root) {{
  source(file.path(project_root, "scripts", "release_guard.R"), local = TRUE)
  review_verify_release(project_root)
}}

{prefix}_require_release <- function(project_root, mode) {{
  if (identical(mode, "canonical")) {prefix}_release_id(project_root)
  invisible(TRUE)
}}

''' + text[end:]
    if rel.endswith('/run_20d_global_null_transformed_canonical.R'):
        start = text.index('if (identical(mode, "canonical")) {\n  release_id <- system2(')
        end = text.index('canonical_common_path <-', start)
        text = text[:start] + '''source(file.path(project_root, "scripts", "release_guard.R"), local = TRUE)
stopifnot(identical(review_verify_release(project_root), release_id))
''' + text[end:]
    if rel.endswith('/run_20d_global_null_canonical.R'):
        marker = 'script_dir <- dirname(script)\n'
        text = replace_once(text, marker, marker + '''project_root <- normalizePath(file.path(script_dir, "../../.."), winslash = "/")
source(file.path(project_root, "scripts", "release_guard.R"), local = TRUE)
stopifnot(identical(review_verify_release(project_root), release_id))
''')
    if rel.endswith('/run_revision_phase.ps1'):
        start = text.index('    $commit = (& git ')
        end = text.index('    return @(', start)
        text = text[:start] + '''    $releaseRoot = Resolve-Path -LiteralPath (Join-Path $repoRoot.Path "..\\..")
    $commit = (Get-Content -LiteralPath (Join-Path $releaseRoot.Path "RELEASE_ID.txt") -Raw).Trim()
    if ($commit -notmatch "^[0-9a-f]{40}$") { throw "Invalid anonymous release ID" }
''' + text[end:]
    # Require the exact anonymous R2 package when a supplied library is selected.
    marker = 'suppressPackageStartupMessages(library(ReviewPkg, lib.loc = normalized))'
    if marker in text:
        text = text.replace(marker, marker + '\n  stopifnot(identical(as.character(utils::packageVersion("ReviewPkg")), "0.0.0.9001"))')
    return text

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve(strict=True); target = args.target.resolve(); audit = args.audit.resolve()
    if target.exists() or target.is_relative_to(source):
        raise SystemExit('Target must be new and outside the source')
    audit.mkdir(parents=True, exist_ok=True)
    with (source/'SOURCE_MANIFEST.csv').open(encoding='utf-8', newline='') as f:
        originals = list(csv.DictReader(f))
    for row in originals:
        data = (source/row['release_path']).read_bytes()
        assert digest(data) == row['release_sha256'], row['release_path']
    target.mkdir(parents=True)
    records = []
    for row in originals:
        old = row['release_path']; rel = old.replace('code/BATTS/', 'code/methods/ReviewPkg/')
        rel = rel.replace('BATTS_functions.R', 'model_functions.R').replace('BATTS-package', 'reviewpkg-package').replace('BATTS_types.h', 'review_types.h')
        if old in {'code/BATTS/README.Rmd'}:
            records.append(dict(source_path=old, release_path='', source_sha256=row['release_sha256'], action='omit redundant public vignette'))
            continue
        data = (source/old).read_bytes()
        if (source/old).suffix.lower() in TEXT:
            s = data.decode('utf-8-sig'); t = neutral(s)
            if old.startswith('code/r2/'):
                t = adapt_workflow(rel, t)
            if old == 'code/BATTS/DESCRIPTION':
                t = (source/'reference/r1/code/methods/ReviewPkg/DESCRIPTION').read_text(encoding='utf-8')
                t = t.replace('0.0.0.9000', '0.0.0.9001')
                t = t.rstrip() + '\nReview-Source-ID: ' + opaque(PACKAGE_SHA) + '\n'
            if old == 'code/BATTS/README.md':
                t = (source/'reference/r1/code/methods/ReviewPkg/README.md').read_text(encoding='utf-8')
                t += '\nRound 2 review source version: 0.0.0.9001. See the release README for isolated installation.\n'
            if old in {'code/BATTS/R/BATTS_functions.R','code/BATTS/R/RcppExports.R','code/BATTS/src/RcppExports.cpp'}:
                r1name = {'code/BATTS/R/BATTS_functions.R':'R/model_functions.R','code/BATTS/R/RcppExports.R':'R/RcppExports.R','code/BATTS/src/RcppExports.cpp':'src/RcppExports.cpp'}[old]
                t = (source/'reference/r1/code/methods/ReviewPkg'/r1name).read_text(encoding='utf-8')
                # The R1 wrappers perform the same operations; prove this below by inverse mapping.
                inverse = t.replace('native_run_tree_model','run_adaboost').replace('native_evaluate_density_ratio_boosting','evaluate_balance_weight_boosting').replace('native_evaluate_density_ratio_bayesian','evaluate_balance_weight_BART').replace('native_rotation_matrix','rotation_matrix')
                inverse = inverse.replace('fit_boosting_model','boots').replace('fit_bayesian_model','batts').replace('evaluate_density_ratio =','eval_balance_weight =')
                inverse = inverse.replace('_ReviewPkg_run_tree_model','_BATTS_run_adaboost').replace('_ReviewPkg_evaluate_density_ratio_boosting','_BATTS_evaluate_balance_weight_boosting').replace('_ReviewPkg_evaluate_density_ratio_bayesian','_BATTS_evaluate_balance_weight_BART').replace('ReviewPkg','BATTS').replace('review_types.h','BATTS_types.h')
                assert inverse.strip() == s.replace('\r\n','\n').strip(), 'R1 wrapper has computational differences: ' + old
            t = t.replace('ReviewPkg_types.h','review_types.h')
            # Preserve bytes of unchanged text, including the historical R1 package.
            if t != s:
                data = t.encode('utf-8')
        dst=target/rel; dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
        records.append(dict(source_path=old, release_path=rel, source_sha256=row['release_sha256'], release_sha256=digest(data), action='identity adaptation' if digest(data)!=row['release_sha256'] else 'unchanged'))
    overlay=HERE/'anonymous_overlay'
    if overlay.exists():
        for f in overlay.rglob('*'):
            if f.is_file():
                dest=target/f.relative_to(overlay);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    shutil.copyfile(HERE/'ACC_form_R2_draft.Rmd', target/'ACC_form_R2_draft.Rmd')
    (target/'RELEASE_ID.txt').write_text(opaque(WORKFLOW_SHA)+'\n',encoding='utf-8')
    (audit/'private_source_mapping.json').write_text(json.dumps(dict(package_commit=PACKAGE_SHA,workflow_commit=WORKFLOW_SHA,package_id=opaque(PACKAGE_SHA),release_id=opaque(WORKFLOW_SHA),files=records),indent=2)+'\n',encoding='utf-8')
    print('Created',target,'with',len(records),'source records; identifying map retained in audit directory')

if __name__ == '__main__': main()
