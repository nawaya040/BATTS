# JASA R2 再現用ファイル群: 読取り調査と実装提案

調査日: 2026-09-24。現在の作業は読取り調査であり、入力データ、原結果、科学計算コード、論文の変更やcanonical計算は行っていない。

## 確認済みの入力

- R1提出アーカイブ: `../density-ratio-revision-inputs/submitted-reproducibility/archive/reproducibility_materials/`。`code/` 2,245ファイル・23.7MB、`data/` 6ファイル・16.0MB、`output/` 605ファイル・31.0MB。`code/r_lib/` はR1時点のローカルインストールで、機種依存の可能性がある。
- R2 canonical保存結果: `C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_backup/results/generated/r2-full-computation-canonical-20260813/`。`bart-transformed/` 404ファイル・14,302MB、`comparators/` 3,200ファイル・259MB、`coverage/` 1,800ファイル・77MB、`summaries/` 24ファイル・34.4MB。`docs/run-handoffs/r2-full-computation-canonical-20260813.md` に計算時のcommit、単位数、ハッシュ記録がある。
- R2従来結果: Dropbox内 `balancePM_results/experiments_1D/` 約0.01GB、`experiments_2D/` 約0.91GB、`experiments_multi/` 約0.29GB。必要な詳細RDSの特定が必要で、ディレクトリ全体を収録しない。
- R2以前のboosting選択結果: Dropbox内 `balancePM_experiment_backup_0806/boosting-selection-full-20260804/` 約278MB。必要な結果とchecksumのみを選ぶ。
- 1D AdaBoost/GB比較結果: Dropbox内 `balancePM_results/experiments_1D/boosting/` 410ファイル・2.1MB。
- 主要なR2集計: `summaries/revision-mse-summary-20260903T170144JST/` に反復別MSE、20D raw/transformed表、CDC stable診断、入力manifest。`summaries/coverage-summary-20260903T170144JST/` に反復別coverage。`summaries/bart-transformed-summary-20260903T170144JST/` にglobal/nullの反復別calibration。`summaries/comparator-summary-20260819T0317JST/` に比較法の反復別指標。
- canonical R2の計算ワークフローのsource commitは `f921525711faef2a326c561820749e0c70f1bf42`（`docs/run-handoffs/r2-full-computation-canonical-20260813.md:7`）。このcommitの `DESCRIPTION` は `Package: balancePM`。別途インストールしたBATTSの指定commit `6f625bad83702b36e5480be1ed1343258a9b075a` は現リポジトリのGitオブジェクトに存在し、その時点の `DESCRIPTION` は `Package: BATTS`（`scripts/revision/20d_global_shift/CANONICAL_README.md:11-12`）。したがって、ワークフローと方法パッケージの2つのsource revisionを別々に記録する。

## 容量と再現の境界

20D BARTの原RDSを全て入れると約14GBになる。提出パッケージには、50反復の図表を正確に再描画・再集計できる検証済みの反復別集計と、代表図に必要な詳細結果を収録する方針を提案する。原計算からの全50反復再実行手順はコードと設定で示し、時間のかかる工程と明記する。light実行は同じseed・シナリオでの計算経路確認とし、木の本数やburn-inを減らす場合には論文数値との一致を主張しない。

R1アーカイブにも2Dの300本、20Dの200本のsummary RDSがあり、図用detail RDSはそれぞれ1本である。R2補足の追加図にはR1 zipにない詳細RDSが必要。各図の原結果の選択とハッシュ照合を終えてから、最終収録リストを固定する。

## 提案する新規ディレクトリと作業単位

提出候補は現コードリポジトリ内の `reproducibility/r2/reproducibility_materials/` に組み立て、原稿・公式結果・R1アーカイブには書き込まない。作業単位は以下の通りとし、別々の差分にする。

1. **無改変のソース・参照結果を選別して収録:** `code/methods/BATTS/` はcommit `6f625bad...` のGit treeから、R2計算ワークフローはcommit `f921525...` と現行の承認済み集計・描画スクリプトから、R1の `ReviewPkg` と修正済み `densratio` はR1アーカイブから、それぞれ由来とSHA-256を記録してコピーする。R1 `code/scripts/`、R2 `scripts/coverage/`、`scripts/boosting/`、`scripts/revision/`、最終図に対応する `scripts/figures/` は必要ファイルを個別に選び、旧版とR2版を分離する。R1の `code/r_lib/`、作業用tmp、launcher logs、候補プレビューは収録しない。
2. **原結果と集計の選別:** R1 `data/`、R1 `output/section34_1d/full/`・`section41_2d/full/`・`section42_multi/full/`・Section 5のCSV、R2の4つの `summaries/` と必要な外部detail RDS・R2比較結果をハッシュ付きで収録する。14GBのBART全RDSを収録しない場合、ACCで50反復の再集計に使う保存済み中間結果を明示する。
3. **再現手順のコードと説明:** `README.md`、図表対応manifest、入力データ辞書、環境・バージョン記録、正確な保存済み結果からの作図・集計手順、補足各図のseedを合わせたlight実行手順を用意する。外部パスを前提とするスクリプトは、入力rootを引数として受ける提出用のコピーまたはwrapperに改め、原コードは変更しない。
4. **検証・梱包:** 原入力とコピーのハッシュ照合、隔離した出力先での各表・図の再集計、論文最終版との比較、light実行の検証を行う。ACCと一致するREADMEを確定してから単一zipを作る。

## 保護対象ファイルの編集ゲート

作業単位1〜3は、`AGENTS.md` が保護する新しいreproducibility code、科学結果のコピー、およびDropboxからのプロジェクト要素のコピーを含む。このユーザーの広い整備依頼は、そのファイル変更に対する承認とは扱わない。実装前に、実際に選ぶ全ファイルと新規コードのパッチを確定し、個別の承認を受ける。ここまでの読取り調査から、最小限必要になり得る既存スクリプトの移植箇所は次の通り。

- `scripts/figures/plot_supplement_s1_r2_preview.R:11-14`: 1D boostingのDropbox絶対パス。
- `scripts/figures/plot_supplement_s2_s3_r2_preview.R:10-14`: 2Dの従来結果、boosting結果、checksumのDropbox絶対パス。
- R1 `code/scripts/section41_2d_common.R` の `get_section41_config()`（約45-90行）: lightはseed 1の1反復・3条件のみ。補足S2/S3に必要な5条件は `scripts/figures/plot_supplement_eight_panel_compact.R:14-18` に列挙され、seedは1、16、21。現行light設定には全条件が揃わない。
- R1 `code/scripts/section42_multi_common.R` の `get_section42_config()`（約60-110行）: lightはseed 21の1条件のみ。補足の図ごとのseedと条件を確認して追加する必要がある。
- R2の描画用プレビューコードには既定出力先が作業ツリー `output/` のものがある。提出用wrapperでは新規の隔離出力先を必須にする。

コピー・引数化・wrapper作成自体では既存の論文数値とcanonical RNGは変えない。lightの新条件は新たな数値を生成するが、論文の代表値として使わず、`SMOKE_NOT_FOR_PAPER` と同等に明示する。検証はコピー前後のSHA-256、parse、隔離先での小規模実行、保存済み結果からの表・図の比較を順に行う。rollbackは新規 `reproducibility/r2/` の候補だけを外し、原入力・結果・論文・既存スクリプトは保持する。

## まだ確定していない事項

- 2D Table 1、CDC診断表に必要なR2原結果と最終作図コードの一対一の対応。補足S2/S3の5設定とS5のseed 21は作図スクリプトから特定したが、原RDSの実在・ハッシュと最終PDFとの一致は未検証。
- 20D反復別集計だけで各図表の独立確認に十分か、代表BART RDSを何本含めるか。
- Section 5のR2最終図の生成経路と共著者担当の入力・公開条件。確認待ち欄をREADMEに設ける。
- zip全体の容量と、JASAアップロード先の実際の上限。

計画時点の見積り: 入力選別と初期パッケージ作成に数時間、表・図の照合とlight実行に追加の時間を要する。full 50反復のcanonical再計算はこの見積りに含めない。
