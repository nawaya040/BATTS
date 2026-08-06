# TEMP: 別PCでのboosting revision作業再開メモ

このファイルは一時的なhandoffメモ。別PCでの再開確認後、独立commitで削除する。

## 1. GitHubのcode repository

Repository:

```text
https://github.com/nawaya040/BATTS.git
```

作業branch:

```text
audit/boosting-selection-diagnostics
```

重要なcommit:

```text
66c74aaa838f07a04141a58ef611c1f5ba0d68ed  canonical runのsource commit
15ca12372bd22a24c7430cb02e8c2ad7589b99c7  700件の最終集計・provenance・checksum
```

通常の`main`上で`git pull`するだけでは、今回の結果bundleは取得されない。
別PCのrepositoryに未commit変更がないことを確認してから、remote名に応じて以下を実行する。

remote名が`github`の場合:

```powershell
git status --short
git fetch github --prune
git switch audit/boosting-selection-diagnostics
git pull --ff-only github audit/boosting-selection-diagnostics
git log --oneline --decorate -5
git merge-base --is-ancestor 15ca12372bd22a24c7430cb02e8c2ad7589b99c7 HEAD
```

remote名が`origin`の場合は、上記の`github`を`origin`に置き換える。
local branchがまだ存在しない場合:

```powershell
git fetch github --prune
git switch --track github/audit/boosting-selection-diagnostics
```

`git merge-base --is-ancestor`の終了codeが0なら、最終結果commitを含んでいる。

GitHub上のcompact result bundle:

```text
results/reference/boosting-selection-full-20260804/
```

主要ファイル:

- `README.md`: 結果、監査、上限到達の説明
- `cell_summary.csv`: 14セルの最終集計
- `upper_bound_hits.csv`: 1,000本上限到達job
- `output_checksums.csv`: 700 result RDSのSHA-256
- `run_provenance.csv`: source・R・package・RNG情報

## 2. Dropboxのraw result backup

Dropbox内の相対path:

```text
balancePM_experiment_backup_0806/boosting-selection-full-20260804/
```

Dropboxでfolder全体を「オフラインで利用可能」にし、download完了後に使用する。
このbackupは原本として保持し、直接分析・編集しない。別PC上の作業用run rootへ
folder全体をコピーしてから使う。

backupの期待値:

```text
original run files: 2,809
result RDS:          700
manifest RDS:        700
logs:                1,400
original run bytes:  290,834,526
```

inventory:

```text
BACKUP_INVENTORY_SHA256.csv
```

inventory自身のSHA-256:

```text
64c8a0c367f65c5426386897d0bb6a3f286fe264cde966876a91885b20dd5dbc
```

backup内の`BACKUP_README.md`にも検証記録とrestore手順がある。

## 3. 別PCでのhash検証

`$backupRoot`を別PC上の実際のpathへ変更して実行する。

```powershell
$backupRoot = "C:\path\to\Dropbox\balancePM_experiment_backup_0806\boosting-selection-full-20260804"
$inventoryPath = Join-Path $backupRoot "BACKUP_INVENTORY_SHA256.csv"
$inventory = Import-Csv -LiteralPath $inventoryPath
$mismatches = @()

foreach ($row in $inventory) {
    $path = Join-Path $backupRoot ($row.relative_path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $path)) {
        $mismatches += "$($row.relative_path): missing"
        continue
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne ([int64]$row.bytes)) {
        $mismatches += "$($row.relative_path): size"
        continue
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $row.sha256) {
        $mismatches += "$($row.relative_path): sha256"
    }
}

"inventory rows: $($inventory.Count)"
"mismatches: $($mismatches.Count)"
$mismatches | Select-Object -First 20
```

期待される出力:

```text
inventory rows: 2809
mismatches: 0
```

## 4. 既知の状態

- canonical gridは14セル×50 seeds、合計700 jobs。すべて完了済み。
- 全700 result/manifest pairは最終監査を通過。
- `job_status.csv`の`FAILED`と`GRID_HAS_FAILURES.txt`はlauncherの終了code取得不具合。
  数値計算の失敗ではない。700 logsすべてにcompletion markerがある。
- Improved AdaBoostはexponential-loss CVで木の本数を選択したReal AdaBoost。
- exponential-loss CVとbalancing-loss CVは全700件で同じAdaBoost ensembleを選択。
- primary AdaBoostの1,000本上限到達は3/700件。Proposed GB/FSは0件。
- 提案法は全7 unbalancedセルと全3 balanced 2Dセルで低い平均MSE。
- Improved AdaBoostは全4 balanced 20Dセルで低い平均MSE。
- 手法間の木の本数はalgorithm固有のiterationなので、主たる性能比較には使わない。

## 5. 何が再開可能か

GitHub branchとDropbox backupがそろえば、以下は再開可能。

- RDSからの再集計
- improved AdaBoostと提案法のtable作成
- reviewer responseの検討
- manuscript修正案の準備
- 既存結果の追加監査

新しい数値実験やcanonical rerunには、追加の環境検証が必要。

- R 4.5.2
- source commit `66c74aaa...`
- `run_provenance.csv`記載のpackage versions
- installed BATTSのfile hashes
- RNG kindとseed mapping
- 外部run rootのpath設定

raw backupには実行時R library全体は含まれていない。既存RDSの読取り・集計には
大きな問題はないが、新規計算では環境を再構築してsmoke testを先に行う。

## 6. 原稿と一時table

code repositoryとpaper/Overleaf repositoryは別。原稿編集を続ける場合は、paper側も
別途同期する。

このPCで作った速報tableはDropbox project内の次のfolderにある。

```text
Rcpp_experiments/active_programs/balancePM_backup/tmp/boosting-full-launcher-20260804/
```

主要table:

- `IMPROVED_ADABOOST_COMPARISON_TABLE_20260806.md`
- `improved_adaboost_comparison_table_20260806.tex`
- `2D_INTERIM_RESULTS_TABLE_20260805.md`

これらは現時点ではGitHub result commitやmanuscriptには未挿入。
