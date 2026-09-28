# 匿名化候補の作成・監査記録

2026-09-28。ユーザーが承認した匿名化プランに基づく再現資料の変更であり、
原稿、原本 ZIP、R1 ソース、既存の公式結果は保持している。
R2 の BATTS ソースを ReviewPkg 0.0.0.9001 として提供し、R1 の匿名 API 名に
そろえた。計算の引数・既定値・乱数順序・コンパイル条件は維持した。
R1 ReviewPkg 0.0.0.9000 は別ソース・別ライブラリとして保持する。

## 対象と検証

提出コピーのソース名、DESCRIPTION/NAMESPACE/Rcpp 登録名、呼出し箇所、
インストール方法、Git 依存のソース確認、個人パス・公開コミット等のメタデータを
匿名化した。七つの RDS は識別用メタデータだけを変え、科学的な内容を同一比較した。
描画コードが固定していたメタデータファイルの SHA-256 も変更後の値へ更新した。
原本との対応情報は `validation_outputs/anonymization-20260928/` に保持し、
提出 ZIP へ含めない。原本との比較結果、ビルドログ、比較用の実行物もこの場所にある。

12 ケースで元 R2 と匿名 R2 の fitted objects、予測、事後サンプル、乱数状態が
完全一致した。数値 CSV の 3,657,717 セル、R2 数値実装・コンパイル設定の 9 ファイル、
R1 パッケージの 24 ファイルを保持した。992 RDS のうち 985 はバイト単位で不変。
表の 264 数値、図 15 点中 14 点が原稿と一致し、Figure 2 の既知の配置差は残る。
匿名 R2 の 2D light 実行は 355.9 秒。完全な依存隔離環境や 50 seed の再実験は未検証。

## 再作成

プロジェクト直下から以下を実行する。`<NEW_CANDIDATE>` と `<NEW_AUDIT>` は
新しいディレクトリを指定し、既存結果を上書きしない。Windows では Rscript の
フルパスを使い、LANG/LC_ALL を C に設定する。匿名化の作成用スクリプトには元の
識別名が含まれるため、これらを提出コピーへ追加しない。

```text
python reproducibility/r2/anonymize_candidate.py --source reproducibility/r2/reproducibility_materials --target <NEW_CANDIDATE> --audit <NEW_AUDIT>
Rscript reproducibility/r2/anonymize_rds_metadata.R reproducibility/r2/reproducibility_materials <NEW_CANDIDATE> <NEW_AUDIT>/rds-content-comparison.csv
python reproducibility/r2/seal_anonymous_candidate.py <NEW_CANDIDATE> --source reproducibility/r2/reproducibility_materials --audit <NEW_AUDIT>/seal.json
python reproducibility/r2/validate_anonymous_content.py --source reproducibility/r2/reproducibility_materials --candidate <NEW_CANDIDATE> --audit <NEW_AUDIT>
Rscript reproducibility/r2/audit_serialized_inputs.R <NEW_CANDIDATE> <NEW_AUDIT>/rds-audit.csv
python <NEW_CANDIDATE>/code/verify_bundle.py
```

`anonymous_overlay/` が提出用変更の編集元、ACC は `ACC_form_R2_draft.Rmd` が編集元。
オーバーレイの検証記録は今回の検証結果なので、将来コードを変えた場合は再検証して
更新する。最終 seal に `--zip <NEW_ZIP>` を付けると全メンバーを再読込して検証する。
提出 ZIP の最上位名は候補ディレクトリ名に一致する。ACC・README の名前も合わせる。
依存バージョンやパッケージ更新を自動で行う作業は含まない。

## 復旧と後続作業

以前の候補と ZIP を保持しているため、比較や復旧ではそちらを参照できる。
匿名化候補の作成・検証は原本へ書き込まない。共著者の Section 5 入力後は
ACC、COAUTHOR_SECTION5.md、manifest、ZIP とハッシュを一緒に更新する。
GitHub/Overleaf への push と共著者への送信は行っていない。


## 共著者共有版

共著者向けの文章整理を適用した後の ZIP とハッシュは `STATUS.md` を参照。
`coauthor_handoff/READ_FIRST.md` が共有フォルダーの案内文の編集元。
共有フォルダーには、案内文、ACC の Rmd/PDF、COAUTHOR_SECTION5.md、
VALIDATION_STATUS.md、再現用 ZIP と SHA-256、全同梱ファイルを対象とする
HANDOFF_SHA256.txt を含める。共有用 ZIP はこのフォルダー全体を圧縮する。
内部監査ログや古い共有版のバックアップは同梱しない。
共有用 ZIP 自体の SHA-256 は `coauthor-review-anonymous.zip.sha256` に記録した。
