> Update 2026-09-28: the active candidate on this PC is now
> `reproducibility_materials_anonymous.zip`, with R2 ReviewPkg 0.0.0.9001 and
> the revised `ACC_form_R2_draft.Rmd`. See `STATUS.md` and `ANONYMIZATION_BUILD.md`.
> The previous archive and the handoff notes below are preserved as history.

# Round 2 ACC: continue on another computer

This handoff records the editable ACC draft and the corresponding technical review package as of 2026-09-28. It is for author and coauthor review. The archive is a review candidate; Section 5 and the items listed in `STATUS.md` still need completion before journal submission.

## Obtain the working files

1. Clone `https://github.com/nawaya040/BATTS.git` and check out `revision/r2-full-computation-prep`. The ACC draft and package records are present from commit `fac23278ee8a4fc51c496bda36bff1f24e740e86` onward. To incorporate later changes, use an ordinary `git pull` on that branch after checking for local edits.
2. Open `reproducibility/r2/ACC_form_R2_draft.Rmd` in the clone. This is the editable Round 2 ACC source. Its Section 5 table and bracketed author fields are intentional completion markers. `reproducibility/r2/overlay/COAUTHOR_SECTION5.md`, `FIGURE_TABLE_MAP.md`, `VALIDATION_STATUS.md`, and `STATUS.md` provide the technical handoff and verification limits.
3. In the same Dropbox account, open `JASA_ACC_work/R2_handoff/reproducibility_materials_review.zip`. Copy it to a local working directory before extracting it. Its SHA-256 is `2625e1c49e8523757b4754b9b106a24bcab0be585b77f9033f2b2dabe1cf5e39`. The archive contains one top-level `reproducibility_materials_review/` directory. Keep the extracted `reference/` and `results/` trees unchanged; direct new outputs to separate directories.
4. For comparison with Round 1, `JASA_ACC_work/ACC_form.pdf`, `acc_form.Rmd`, and `reproducibility_materials.zip` are already in Dropbox. The Round 1 PDF SHA-256 is `3ffdaf03673db57f18cb0edc43e1b341a898f2900f6d099bd5ca7eec7e4750aa`.

The copies in the local Dropbox folder were hash-checked after placement. Dropbox cloud synchronization could not be independently checked through the web account on this computer; confirm that `R2_handoff/` appears on the second computer before relying on it as the sole copy.

On Windows PowerShell, check the downloaded review archive with `Get-FileHash -Algorithm SHA256 -LiteralPath <path-to-zip>`. From the extracted package directory, run `python code/verify_bundle.py` to check its file manifest. This verifies package integrity; it does not rerun the paper's analyses.

## Source and storage boundaries

GitHub contains the ACC source, assembly scripts, release-specific code, maps, and verification records. The assembled archive, copied input data, saved scientific results, and local validation outputs are excluded from Git. The Dropbox archive supplies the review package without publishing the processed Section 5 data to GitHub. The complete transformed 20D BART raw RDS collection (approximately 14 GB) is outside this 48 MB archive; its seed-level summaries, source code, and provenance are included. A full canonical recomputation requires the separately retained raw result/data roots and a prepared R environment.

The current manuscript is maintained in the separate Overleaf-connected `density-ratio-paper` repository. Use the authoritative paper version there when checking table and figure references; this ACC handoff does not alter or push the paper. The prior Round 1 manuscript PDF in Dropbox is historical.

## Work remaining before the final ACC PDF

- Have the Section 5 coauthor confirm final Round 2 inputs, plotting commands, data lineage, versions, and sharing rights. Update the package and the corresponding ACC text together.
- Resolve the Main Figure 2 layout difference, pin and check the dependency environment, and measure the complete reviewer workflow and resource use. `VALIDATION_STATUS.md` details the current evidence.
- Confirm the final archive name and hash, complete the checkboxes and bracketed author fields in `ACC_form_R2_draft.Rmd`, then render and inspect the PDF for submission.

The Dropbox `R2_handoff/` files are copies for transfer. Treat the GitHub ACC draft as the editable source; replace the Dropbox copies only after the new package and hash are verified.
