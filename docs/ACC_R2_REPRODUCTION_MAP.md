# JASA Round 2: 図表と再現用ファイルの対応表

作成日: 2026-09-24。これは提出物を組み立てる前の調査表であり、再実行や数値照合の完了記録ではない。対象は `../density-ratio-paper/JASA_submission_R2/main_JASA_Aug10_2026.tex` と `supplement_JASA_Aug10_2026.tex` の現行ローカル版。論文内の図番号と画像ファイル名は一致しない場合があるため、表ではキャプションの内容も記した。原稿が変わった場合は、この表を先に更新する。

## 表の読み方

- **R1**: 前回提出した `../density-ratio-revision-inputs/submitted-reproducibility/archive/reproducibility_materials/` に入力、コード、結果、または手順がある。「従来のRDS」のうち、このアーカイブに存在しないものは別途特定が必要。
- **R2**: 現リポジトリにRound 2用の計算・集計・描画コード、または候補出力がある。
- **要追跡**: 論文画像は確認できたが、最終画像を生成した入力とコードの全連鎖は、この調査でまだ確認できていない。
- **共著者**: Section 5の内容確認・追記が必要。Codexは技術的な照合と下書きを担当する。

画像が論文リポジトリに存在することだけでは再現性の検証にならない。保存済みの `output/figures/` と `output/pdf/` にはプレビューや候補も含まれるため、提出する画像との同一性を別途確認する。以下の「次の確認」は、ACCの再現範囲とzipの収録物を確定するための作業である。

## R1提出物で確認できた再現範囲

前回ACC（`C:/Users/user/Downloads/ACC_form.pdf`）は「選択した図表」を対象とし、Section 3.4の1D図、Section 4.1のTable 1・Figure 3、Section 4.2のTable 2・Figure 4、Section 5の実データ図を列挙している。READMEも主としてこれらの本文結果の手順を説明する。補足の全図表を再現対象とした記述はない。1Dの4設定の図とその入力RDSは入っているため一部の補足図には利用できるが、2D補足の5枚について詳細RDS・個別描画手順がすべて揃う構成ではない。20Dの変換後設定は実行オプションがある一方、full結果は同梱されていないとREADMEとACCが明記する。

R1アーカイブのfull出力には、2Dの6セル×50反復のsummary RDS（計300本）と本文Figure 3用のdetail RDS（1本）、20Dの4セル×50反復のsummary RDS（計200本）と本文Figure 4用のdetail RDS（1本）がある。表は保存済みの50反復から再集計し、図は指定反復（2D: seed 1、20D: seed 21）の詳細出力から再描画する構成である。`light` は2Dで1反復、20Dでseed 21の1反復を走らせるが、木の本数・burn-in等も減らすため、論文図表との数値一致を目的としない。`full` の50反復再実行コマンドは用意されているものの、短時間で確認する主手順は同梱済みfull出力の再集計・再描画とlight実行である。

## 本文: シミュレーションと方法評価

| 論文の結果・画像 | 確認した入力・生成経路 | 現状 | 次の確認 |
|---|---|---|---|
| Figure 1: 1Dの事後分布とcoverage（`figure1_r2_balanced_sample_size.pdf`） | R1 `output/section34_1d/full/`。R2 `scripts/figures/plot_figure1_r2_preview.R` は `config.dput`、`coverage/`、`illustration/` を読む。 | R1+R2 | 本文の均衡2設定に対応する最終入力とPDFのハッシュを確認。R1の古いcoverageとR2修正後のcoverageを区別する。 |
| Figure 2: 2Dの生成例（`figure2_r2.pdf`） | R2 `scripts/figures/plot_figure2_r2_preview.R`、`scripts/coverage/models/section41_2d_models.R`。 | R2 | 描画時のseed・設定と論文PDFを照合し、生成例を再作図する手順を確定。 |
| Table 1: 2DのMSEとSE（TeX表） | R1 `output/section41_2d/full/` と `code/scripts/summarize_section41_2d_table.R`。R2ではAdaBoost DRT・CDC等の値が更新されている。 | R1+R2、要追跡 | 各方法・6セル・50反復の原結果から表の全数値への経路を追跡。特にR2値とCDCのfinite-only規則を確認。 |
| Figure 3: 2D local shiftの推定面（`figure3_compact_r2.pdf`） | R2 `scripts/figures/plot_figure3_compact_preview.R`。R1系の詳細RDS、R2のboosting RDSとchecksum、2Dモデルを読む。候補の `output/figures/figure3-compact-preview/validation.txt` は入力ハッシュを記録。 | R1+R2 | 指定反復・設定、最終PDFと候補PDFの同一性、RDSの収録可否を確認。 |
| Table 2: 20DのMSEとSE（TeX表） | R2 `scripts/revision/summarize_revision_mse.R` が従来の20D RDS、R2 boosting・kernel・CDC、global/null BARTの集計を結合し、`mse_table_20d_raw.csv` を書く。 | R1+R2 | 論文の全セルをCSVと照合。入力manifest・参照結果の所在と収録範囲を確定。 |
| 20D calibration図（`figure20d_calibration_compact_r2.pdf`） | R2 `scripts/revision/20d_global_shift/summarize_20d_global_null_canonical.R` の `seed_calibration_curves.csv`、`scripts/coverage/summarize_coverage.R` の `coverage_by_seed.csv`。最終候補は `scripts/figures/plot_figure4_compact_preview.R`。 | R2 | 50反復×8パネルの入力CSVと最終PDFを照合。計算結果と表示専用コードをともに収録。 |
| 20D pointwise evaluation図（`figure20d_pointwise_evaluation_compact_r2.pdf`） | R2 `scripts/figures/plot_figure_20d_localization_r2_preview.R` は従来のlocation/dispersion RDSと新しいglobal RDSから `figure_20d_localization_r2_preview_summary.csv` を作る。`plot_figure5_compact_preview.R` が表示する。 | R1+R2 | 10反復の原RDS、CSV、最終PDFの対応とハッシュを確認。 |

## 補足: シミュレーション、診断

| 論文の結果・画像 | 確認した入力・生成経路 | 現状 | 次の確認 |
|---|---|---|---|
| 1D AdaBoost対GB図（`figureS1_r2.pdf`） | R2 `scripts/figures/plot_supplement_s1_r2_preview.R`。50反復のboosting出力を読み、入力一覧CSVを作る。現スクリプトにはDropbox絶対パスがある。 | R2 | 50反復の実ファイルを特定し、最終PDFと照合。提出用の相対パス化は別途パッチ承認が必要。 |
| 2D追加推定面5枚（S2の2枚、S3の3枚） | R2 `scripts/figures/plot_supplement_s2_s3_r2_preview.R` と `plot_supplement_eight_panel_compact.R`。R1時代の外部詳細RDS、R2 boosting RDS、checksumを使用。R1提出アーカイブにはこの5枚の詳細RDSは見当たらない。 | R2、要追跡 | `figureS2_*_compact_r2.pdf`、`figureS3_*_compact_r2.pdf` の各画像と外部入力RDSを個別に対応付ける。前者のスクリプトにはDropbox絶対パスがある。 |
| 2D coverage図（`figureS4_compact_r2.pdf`） | R2 `scripts/coverage/summarize_coverage.R` の `coverage_by_seed.csv`、`coverage_by_nominal_mass.csv`、`summary_metadata.txt`。`scripts/figures/plot_supplement_s4_compact_preview.R` が描画。 | R2 | corrected coverageの50反復・6セルを確認し、論文PDFと照合。 |
| CDC数値診断表（TeX表） | 本文Table 1と同じCDC原結果に加え、安定版計算の50反復が必要。補足TeXにはfinite反復数とstable MSE/SEが記載されている。 | R2、要追跡 | 当該4セルの原出力、安定版集計コード、表の数値を一対一で照合する。 |
| 20D locationの推定面（`figureS5_latent_location_compact_r2.pdf`） | R2 `scripts/figures/plot_supplement_eight_panel_compact.R`。R1時代の外部20D詳細RDS、R2 boosting RDS、モデル・checksumを使用。R1提出アーカイブのdetail RDSはseed 21で、この図の入力と同じか未確認。`output/figures/supplement-eight-panel-final-preview/s5_latent_location_balanced_validation.txt` に入力ハッシュがある。 | R2、要追跡 | 最終PDFと候補、外部詳細RDS、boosting RDSを照合。 |
| 変換後20Dの生成例（`figureS6_compact_r2.pdf`） | R2 `scripts/figures/plot_supplement_s6_compact_preview.R` と20Dモデル。記録された設定はseed 2、各群500、変換あり。 | R2 | 同じモデル・設定で再描画し、論文PDFと照合。推定器の再実行は不要。 |
| 変換後20DのMSE表（TeX表） | R2 `scripts/revision/summarize_revision_mse.R` の `mse_table_20d_transformed.csv`。 | R1+R2 | すべての方法・セル・SEをTeX値と照合。 |
| 1D不均衡の事後分布とcoverage（`figureS6_1d_unbalanced_sample_size_r2.pdf`） | R1 `output/section34_1d/full/`。R2 `scripts/figures/plot_figure1_r2_preview.R` の不均衡presetが候補。 | R1+R2 | 本文Figure 1と入力系列が同じか、corrected coverageか、最終PDFの生成条件を確認。ファイル名のS6は図番号を確定する証拠にしない。 |

## 本文・補足: Section 5 application（共著者確認欄）

R1には `data/sample_train.csv`、`sample_test.csv`、4生成法の `sample_*.csv`、`output/revision/{train,test}/log_w_*.csv`、`code/plot_revision1.py`、`code/scripts/run_section5_biodata.R` がある。これらは前回の提出物であり、Round 2の最終画像と同じ入力・処理であることは未確認。以下の各行について、Codexが技術的に照合した内容をACCとREADMEの下書きに記し、共著者が由来・解釈・公開条件を確認する。

| 論文の結果・画像 | 暫定的な対応 | 共著者に確認する事項 |
|---|---|---|
| 本文Figure 5: test PCoA（`figure5_test_ggplot2.pdf`） | R1の処理済みtestデータと4種類の生成サンプル。最終PDFを作ったRスクリプトは未特定。 | Bray–Curtis/PCoAの実装、生成サンプルの版、再描画コマンド。 |
| 本文test DRE図（`test_ggplot2.pdf`） | R1のtest `log_w_*.csv` と処理済みデータが候補。最終PDFの生成コードは未特定。 | 推定RDS/CSVの由来、CI・top 20選択規則、再描画コマンド。 |
| 補足train PCoA（`figure5_train_ggplot2.pdf`） | R1のtrainデータと生成サンプルが候補。最終PDFの生成コードは未特定。 | test版と同じ前処理・PCoA設定か。 |
| 補足train DRE図（`train_ggplot2.pdf`） | R1のtrain `log_w_*.csv` が候補。最終PDFの生成コードは未特定。 | 推定結果、CI・top 20選択、描画コード。 |
| 補足全例DRE図2枚（`supp_PCoA_DRE_train.png`、`supp_PCoA_DRE_test.png`） | R1 `code/plot_revision1.py` と `code/utils/` が候補。 | 画像を作った正確なコマンド、入力ファイル、Python環境。 |
| 補足null実験図（`null_test_credible_bands.pdf`） | R1のREADMEとACCにはこのR2図の明示的な手順がない。 | null実験の入力、推定コード、292 test観測の出力、図の生成コマンド。 |

Section 5の共通確認事項: 元データ `curatedMetagenomicData` からの取得・前処理と処理済みCSVの関係、生成法ごとのコードと版、データ辞書、権利・公開先、R1からの変更点。R1 ACCは処理済みCSVからの再現範囲を明示し、元データからの全工程を再現するとは記していない。R2でも実際に提供する範囲を正確に記載する。

## 提出物を確定する前の優先作業

1. **表と原結果の数値照合:** 本文Table 1・Table 2、補足の2表を最優先とする。図の見た目だけでは数値系列の一致を確認できない。
2. **R2原結果の所在とハッシュ:** `docs/run-handoffs/r2-full-computation-canonical-20260813.md` に記録されたcommit・グリッド・結果ハッシュを、実ファイルに対して検証する。記録だけをもって現PCに全ファイルがあるとは扱わない。
3. **最終画像の同定:** 論文 `figures/` と現リポジトリのpreview/candidate PDFを比較し、どのスクリプト・入力・版が最終版か確定する。
4. **収録範囲・サイズ:** 保存済みの全反復をzipに入れるか、監査可能な集計・参照出力と再計算コードを入れるかを図表ごとに決める。省く場合はACCで正当化する。R1のローカル `code/r_lib/` は機種依存の可能性があるため、含める判断を再検討する。
5. **実行可能な手順:** Dropbox絶対パスが残る描画コードと外部結果ルートの指定を整理し、隔離先で実行する。保護対象コードのパス変更には、正確な差分・数値/RNG影響・検証/復帰計画を提示したうえで個別承認を受ける。
6. **Section 5の引継ぎ:** 共著者の確認欄を埋め、未確認のデータ・図・公開条件をACCで完成済みと表現しない。

本表の作成では、科学コード、論文、入力データ、既存結果を変更していない。canonical実験も実行していない。
