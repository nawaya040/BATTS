# JASA R2 ACC: このPCでの受入れ確認

調査日: 2026-09-28。対象はGitHubコミット `9f14790ba4df1bd7a30dc5f9fcc7bfdd4a20c94e`
とDropbox `JASA_ACC_work/R2_handoff/reproducibility_materials_review.zip`。
原稿、科学計算コード、ACC草案、受領ZIPの内容は変更していない。

## GitHub・Dropbox・原稿の対応

以前の作業ブランチは `audit/boosting-selection-diagnostics`、HEADは
`d8df832`（2026-08-06）だった。作業ツリーに未コミット変更はなかった。
R2ブランチとの比較は旧側32コミット、R2側17コミットで、共通祖先は存在しない。
旧側のrootは `b1d0434`、R2側のrootは `14f9f5a` であり、単純な遅れの関係ではない。
ファイル差分は169ファイル、23,835行追加、1,842行削除。
R2側から `codex/r2-acc-review-20260928` を作成してこのPCの作業先とした。
旧ブランチ、外部runディレクトリ、Dropbox原本は保持している。

旧ブランチはBATTSパッケージ、R2ルートはbalancePM研究ワークフローである。
再現ZIPに収録されたBATTSは `6f625bad83702b36e5480be1ed1343258a9b075a`、
canonicalワークフローの記録は `f921525711faef2a326c561820749e0c70f1bf42`。
この2種類のソースを区別して環境を準備する必要がある。

DropboxのACC草案とGitHubの草案はバイト単位で一致し、SHA-256は
`73a3f8339ef81e20e673acb5e4fbc0a4aae6a66ba1e8c293fa0a1c4e50e86e97`。
ZIPは48,331,384 bytes、SHA-256は
`2625e1c49e8523757b4754b9b106a24bcab0be585b77f9033f2b2dabe1cf5e39`。

Overleafプロジェクト `6075123cfc93d6e6a86a2dee` は当初アプリ内ブラウザーで
閲覧権限エラーを返した。その後、ユーザー指定のGit URLから
`density-ratio-paper/` に独立リポジトリとしてクローンした。
`main` と `origin/main` は `ddaae38c2ddcf296a55cd8bb4d6b19956be892fe`
（2026-09-24 07:37:02 UTC）で一致し、原稿作業ツリーはクリーン。
親の `.gitignore` で原稿リポジトリを除外した。原稿の編集・pushは行っていない。
別PCの「PDF 14点一致」は引継ぎ資料の記録として保持し、今回の確認と区別する。

## このPCで実行した検証

受領ZIPのCRC、メンバー一覧、manifest記載ファイルのSHA-256を検証した。
1,268メンバーは1,267個のmanifest対象とmanifest自身に一致する。
展開後の `code/verify_bundle.py` は成功し、112 MSE設定・5,600行、
72 coverage設定・356,400行のseed集合1–50、7点の詳細RDS、選択されたboosting結果を確認した。
この検証は保存結果の整合性確認であり、推定器の全再実行を意味しない。

`code/portable/reproduce_tables.py` でMain Table 1、Main Table 2、
transformed 20D補足表、CDC診断表を再集計した。行数は42、42、42、4。
Gitに保存された `overlay/verification/tables/` と全CSVフィールドが一致した。
クローン後、同梱の `compare_tables_to_tex.py` を使い、
`JASA_submission_R2/main_JASA_Aug10_2026.tex` と
`supplement_JASA_Aug10_2026.tex` に対して直接照合した。
264項目の表示数値が一致し、比較プログラムは正常終了した。

展開先 `code/` のRソース54本とPythonソース3本は構文解析に成功した。
この確認には `reference/r1/` 全体の実行や全科学計算の妥当性検証は含まれない。

Main Figure 5の保存集計からの描画は、このPCのR 4.5.2で成功した。
Main Figure 2は当初のggplot2 4.0.0 / patchwork 1.3.0で
`Can't find method for generic &(e1, e2)` と停止した。別PCで検証された依存バージョンを
専用ライブラリに準備したところ再試行は成功し、3シナリオ・30,000観測のPNG/PDFを生成した。
PNGを目視して3パネル、軸、凡例を確認した。最終原稿とのレイアウト差の判定は未実施。

2D local-shift、各群5,000、seed 1のlight実行は368.2秒で正常終了した。
7手法のpooled/group別誤差が有限であること、詳細データが10,000行×2列であること、
詳細推定値が有限でcoverageが[0,1]内にあることを確認した。
40本のBayesian trees、40 burn-in、80 backfittingの縮小設定であり、
`SMOKE_NOT_FOR_PAPER=true` を出力している。全50反復の結果一致は検証していない。
実行途中に観測したR workerのpeak working setは約584 MiBだったが、
これは処理全体の最終ピークやreviewerワークフロー全体の資源測定には相当しない。
20D light実行は今回未実施。

詳細な記録はGit対象外の `reproducibility/r2/validation_outputs/pc-20260928/` にある。
`integrity.json`、`table_comparison.json`、`section5_input_inventory.csv` に
転送検証、表の比較、応用データの形状とハッシュを保存した。

## 環境

R 4.5.2、Python 3.14.4、Rtools45 / GCC 14.3.0を確認した。
既存Rライブラリの主なバージョンはggplot2 4.0.0、patchwork 1.3.0、pracma 2.4.4、
scales 1.4.0、mvtnorm 1.3-3、ada 2.0-5、Rcpp 1.1.1-1、RcppArmadillo 14.4.1-1。
通常のライブラリにReviewPkgは入っていなかった。

ZIPのReviewPkgと修正版densratioのソースを検証用buildディレクトリにコピーし、
専用Rライブラリへのビルド・ロード確認を完了した。制限環境では子プロセスのPATHに
g++が見つからず、通常環境での再試行により成功した。ソースの編集は行っていない。
専用ライブラリにはggplot2 4.0.3、patchwork 1.3.2、pracma 2.4.6も用意した。
これらは引継ぎ記録のバージョンと一致する。その他の依存は既存ライブラリを参照するため、
依存の完全な固定、空のライブラリだけを用いる確認、ピークメモリ計測は別途必要。

## 共著者が確認すべきSection 5の具体的な状態

受領ZIPには入力CSVが6本ある。`sample_train.csv` は1,166行×123列、
`sample_test.csv` は292行×123列、`sample_d.csv`、`sample_dt.csv`、
`sample_icfm.csv`、`sample_mbgan.csv` は各1,000行×123列。
保存log-ratioはtestが各1,292行×10列、trainが各2,166行×10列で、各4手法分がある。
列の生物学的意味、単位、正規化、欠損値処理、元データ版、分割の根拠は担当者の確認が必要。

R1の `reference/r1/code/plot_revision1.py` は6つの入力CSVと保存log-ratioを読み、
`plt.show()` を呼ぶ対話的描画コードで、`savefig` 呼出しを含まない。
最終R2図のPDFを一括保存するコマンドとしては未確定である。
`run_section5_biodata.R` の存在と保存CSVの存在だけから、全R2応用図の生成経路を確定できない。
共著者には既存 `COAUTHOR_SECTION5.md` の6行に、主論文test PCoA/DRE、
補足train PCoA/DRE、全標本DRE、null実験について、最終入力・コード・コマンド・
ソフトウェア版・確認者を記入してもらう。

データ配布条件とコードライセンス、GPUの要否、所要時間、出力図の照合も必要。
今回、応用データの再推定、共著者への送信、外部へのアップロードは行っていない。

`validation_outputs/pc-20260928/coauthor-review/` にACC草案、受領ZIP、
Section 5依頼表、図表対応表、別PCの検証記録の無改変コピーと英語の `READ_FIRST.md` を用意した。
コピー5点のハッシュ一致を検証し、`TRANSFER_CHECKSUMS.json` に記録した。
共著者の記入を受けるためのレビュー用一式として利用できる。

## 提出前の修正・確認候補

1. ZIP内部名の説明を統一する。`make_review_zip.py` は最上位名を
   `reproducibility_materials/` に固定している一方、ACC、STATUS、引継ぎメモは
   `reproducibility_materials_review/` を記載している。現物に説明を合わせる方法と、
   梱包コードを変更して再梱包・ハッシュ更新する方法がある。今回は現物を保持した。
2. Main Figure 2の最終原稿とのレイアウト差を確認する。別PCの記録は110 dpiで17.3%の
   pixel差。依存の互換性エラーと、この既知の図面差は別々に検証する。
3. ACCのdata dictionary欄はチェック済みだが、参照文書の多くはファイル対応表である。
   Section 5の変数・単位の辞書が充足しているか、担当者に確認する。
4. 全再計算の入口と必要な外部ファイルを検証する。約14 GBのtransformed BART RDSは
   ZIP対象外。保存集計からの再描画、light試行、canonical再計算の範囲をACCで明確に保つ。
5. 完全な環境記録、reviewer手順全体の時間・メモリ、最終ZIP名・ハッシュ、権利・
   ライセンスと認証チェック欄を確定した後に、ACC PDFを作成・目視確認する。

科学コードや梱包コードの変更、原稿修正は今回適用していない。
今後保護対象を変更する際は `AGENTS.md` に従って具体的な差分と検証方法を提示する。


## 匿名化の実行結果（2026-09-28）

承認された計画に沿って匿名 R2 ReviewPkg 0.0.0.9001 を作成し、ACC を改訂した。
作成手順と監査記録は `reproducibility/r2/ANONYMIZATION_BUILD.md`、現候補の
検証範囲は `reproducibility/r2/anonymous_overlay/VALIDATION_STATUS.md` を参照。
ZIP は 1,284 メンバー、49,369,648 bytes。SHA-256 は
`fab44b62d7bd56c6c8b5dc038b9740bdb2e7932774219c6172d21e4b72859ea9`。

元 R2 との 12 比較は推定物・予測・事後サンプル・乱数状態まで完全一致。
264 表数値と技術図 15 点中 14 点が原稿と一致した。Figure 2 の既知の配置差、
Section 5 の共著者確認、依存環境の完全隔離テストと資源計測は残っている。
再現用コードのソース整合性確認、異なる R1 ライブラリの誤使用拒否も確認した。
原本 ZIP と Overleaf 作業ツリーは保持し、外部への送信・push は行っていない。
