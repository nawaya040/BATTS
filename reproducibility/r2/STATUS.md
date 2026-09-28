# Round 2 reproduction package status

Updated 2026-09-28. The active author-review candidate is
`reproducibility_materials_anonymous/`, with archive
`reproducibility_materials_anonymous.zip` and adjacent SHA-256 record.
It contains anonymous R2 ReviewPkg 0.0.0.9001, the unchanged historical R1
ReviewPkg 0.0.0.9000, reproduction commands, saved scientific results, the
revised ACC source, and validation records. The two method versions use
separate local libraries.

The earlier received archive `reproducibility_materials_review.zip` is preserved
unchanged (SHA-256 `2625e1c49e8523757b4754b9b106a24bcab0be585b77f9033f2b2dabe1cf5e39`).
On this PC it is extracted under its actual ZIP root, `reproducibility_materials/`.
Earlier notes about a removed staging directory concern the other PC.

The editable form is `ACC_form_R2_draft.Rmd`; its current review PDF is
`../../output/pdf/ACC_form_R2_draft.pdf`. Coauthor handoff copies are under
`validation_outputs/anonymization-20260928/coauthor-review-anonymous/`.
Section 5 lineage, permissions, final application commands, a fully isolated
dependency check, resource measurements, and the known Main Figure 2 layout
difference remain open. Final submission certification is intentionally pending.

See `anonymous_overlay/VALIDATION_STATUS.md` for current checks and
`ANONYMIZATION_BUILD.md` for private assembly/audit instructions.

## Coauthor sharing archive

The reviewed sharing archive is
`validation_outputs/anonymization-20260928/coauthor-review-anonymous.zip`
(49,154,391 bytes; SHA-256
`138b6b7b255c6a291e300f427d5e05f32ffd03112f3989b3e86d74948d305914`).
It includes the revised ACC PDF/source, coauthor instructions, validation status,
the reproduction ZIP, and checksum records. Its README source is tracked in
`coauthor_handoff/READ_FIRST.md`. ZIPs, rendered PDFs, private mappings, and local
R libraries are retained locally and excluded from Git.

The current inner reproduction ZIP SHA-256 is
`1ce8f89710d9da3f8c73d677fb3b32844895cdb65a854ca683c0b3356259174a`.
Earlier archive hashes in dated audit notes identify earlier versions.
The final editorial changes affect documentation, code comments, and displayed
text only; 1,193 scientific input/result and package files remained unchanged.
