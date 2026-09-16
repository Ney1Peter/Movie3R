# Local paper and visualization assets

These files are deliberately not mirrored to the code repository. Keep only
the current editable source and, where useful, one immediately preceding
version.

## Retained editable sources on the frozen server

```text
/data/wangzheng/iJCV-CODE/Shot3R_pipeline_open_layout_v072_aligned_20260916.pptx
/data/wangzheng/iJCV-CODE/Shot3R_pipeline_open_layout.pptx
/data/wangzheng/iJCV-CODE/Shot3R_teaser_cropped_enlarged_editable_v9.pptx
/data/wangzheng/iJCV-CODE/Shot3R_alignment_module_redesigned_editable_20260910.pptx
/data/wangzheng/iJCV-CODE/Shot3R_qualitative_3rows_reordered_emphasis_editable.pptx
```

The current paper workspace is:

```text
/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v072_20260916_registration_controls
```

The prior stable local snapshot retained for comparison is v067. Intermediate
paper versions and rendering/build caches are not required for recovery.

The retained EgoHumans visualization workspaces and the currently active
viewer are indexed in
`publication/EGOHUMANS_VISUALIZATION_CATALOG_20260916.md`. The catalog is
tracked as provenance; its large rendering payloads remain local.

## Regeneration policy

- PDF previews can be regenerated from PPTX and should not be retained beside
  every editable source.
- Raw visualization payloads and viewer caches are excluded unless they are
  the only source of a figure.
- Final paper figures already embedded in the retained paper workspace do not
  need a second copy in the code repository.
- Provenance manifests and selection lists are retained in Git even when their
  large image/mesh payloads are removed.
