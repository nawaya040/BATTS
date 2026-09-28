# このPCでのJASA R2 / ACC確認

## 2026-09-28 匿名化後の作業入口

現在のレビュー候補は `reproducibility/r2/reproducibility_materials_anonymous.zip`。
ACC は `reproducibility/r2/ACC_form_R2_draft.Rmd`、閲覧用 PDF は
`output/pdf/ACC_form_R2_draft.pdf` にある。匿名 R2 ソース ReviewPkg 0.0.0.9001 と
R1 0.0.0.9000 を分け、R2 の新しいライブラリは
`reproducibility/r2/validation_outputs/anonymization-20260928/library-r2/` に作成した。

共著者に渡すコピーは
`reproducibility/r2/validation_outputs/anonymization-20260928/coauthor-review-anonymous/`。
同フォルダーの `READ_FIRST.md` が入口になる。受領 ZIP と以前のレビュー用コピーは
比較用に保持している。以降の初期設定記録にある受領候補・R1 light 実行は履歴として読む。
今回の詳細は `reproducibility/r2/ANONYMIZATION_BUILD.md` と
`reproducibility/r2/anonymous_overlay/VALIDATION_STATUS.md` に記録した。

2026-09-28に、GitHubの `revision/r2-full-computation-prep` のコミット
`9f14790ba4df1bd7a30dc5f9fcc7bfdd4a20c94e` を取得し、
`codex/r2-acc-review-20260928` を作業ブランチとして用意した。
以前の `audit/boosting-selection-diagnostics` はそのまま保持している。

## 作業の入口

| 用途 | 場所 |
|---|---|
| 編集元のACC草案 | `reproducibility/r2/ACC_form_R2_draft.Rmd` |
| このPCでの確認結果と残課題 | `docs/ACC_R2_PC_REVIEW_20260928.md` |
| 共著者に記入を依頼する項目 | `reproducibility/r2/overlay/COAUTHOR_SECTION5.md` |
| 別PCからの引継ぎ記録 | `reproducibility/r2/CROSS_PC_ACC_HANDOFF.md` |
| 受領ZIPのローカルコピー（Git対象外） | `reproducibility/r2/reproducibility_materials_review.zip` |
| 受領ZIPの展開先（Git対象外） | `reproducibility/r2/reproducibility_materials/` |
| このPCの検証出力・ログ・Rライブラリ（Git対象外） | `reproducibility/r2/validation_outputs/pc-20260928/` |
| Overleaf連携の独立Gitリポジトリ（親Gitの対象外） | `density-ratio-paper/` |
| R2主原稿・補足・回答書 | `density-ratio-paper/JASA_submission_R2/` |

**ZIP内部のフォルダー名に注意:** 受領したZIPの実際の最上位ディレクトリは
`reproducibility_materials/`。引継ぎメモとACC草案にある
`reproducibility_materials_review/` とは異なる。このPCの展開先には実物の名前を採用した。
`STATUS.md` に登場する別PCの削除済み旧stagingと、このPCで今回新たに展開したコピーは別物である。

## このPCでの実行方法

PowerShellをプロジェクト直下で開く。RscriptはPATHに登録されていないため、
実行ファイルを指定する。専用ライブラリを先に参照し、足りない依存は既存ライブラリから
読み込む構成である。完全に隔離された依存環境としての認証はまだ行っていない。

```powershell
$repo = (Get-Location).Path
$rscript = Join-Path $env:ProgramFiles 'R\R-4.5.2\bin\Rscript.exe'
$bundle = Join-Path $repo 'reproducibility\r2\reproducibility_materials'
$audit = Join-Path $repo 'reproducibility\r2\validation_outputs\pc-20260928'
$env:R_LIBS_USER = (Join-Path $audit 'library') + ';' + (Join-Path $env:LOCALAPPDATA 'R\win-library\4.5')
$env:LANG = 'C'
$env:LC_ALL = 'C'
Push-Location $bundle
python code/verify_bundle.py
Pop-Location
```

再集計は `python code/portable/reproduce_tables.py <新しい出力ディレクトリ>`。
図とlight実行の引数は展開先の `README.md` を参照し、Rscriptには上記の
`& $rscript --vanilla` を使う。毎回新しい出力先を指定する。実行済みの成果物は
`validation_outputs/pc-20260928/` に残してある。

このブランチのルートは `Package: balancePM` の研究ワークフローであり、R2実験に指定された
BATTSソースは展開先の `code/BATTS/` にある。ルートをBATTSとしてインストールしない。
light実行は同梱R1ソースからビルドした `ReviewPkg` と修正版 `densratio` を使う。

## 共著者への引継ぎ

レビュー用に渡す候補はACC草案、受領ZIP、`COAUTHOR_SECTION5.md`、このPCでの確認結果。
ACC草案とZIP等の無改変コピー、英語の依頼事項、転送ハッシュを
`reproducibility/r2/validation_outputs/pc-20260928/coauthor-review/` にまとめた。
同ディレクトリの `READ_FIRST.md` が共著者向けの入口になる。
ZIPには処理済み応用データが含まれるため、既存の共同研究用経路で共有する。
GitHubへのデータ追加や共著者への送信は今回行っていない。

Section 5担当者には、最終図に対応する入力・実行コマンド、データ取得と前処理、
データ辞書、配布条件、Python/Rの依存・所要時間を確認してもらう。記入箇所は
ACCのPart 1、Part 2の環境・ライセンス、Part 3のSection 5表と所要時間にまたがる。
未確定項目を残した著者レビュー用資料として扱い、提出用の認証チェックは保留する。

Overleafはユーザー指定のGit URLから `density-ratio-paper/` にクローン済み。
`origin` は `https://git@git.overleaf.com/6075123cfc93d6e6a86a2dee`、ブランチは `main`、
取得時HEADは `ddaae38c2ddcf296a55cd8bb4d6b19956be892fe`。
主原稿は `JASA_submission_R2/main_JASA_Aug10_2026.tex`、補足は
`JASA_submission_R2/supplement_JASA_Aug10_2026.tex`。
再集計した4種類の表の264数値はこのTeXと一致した。図の照合は別途行う。
原稿は独立リポジトリとして親の `.gitignore` で除外している。
更新確認には `git -C density-ratio-paper fetch origin` を使い、原稿変更やpushは
`AGENTS.md` の承認手順に従う。
