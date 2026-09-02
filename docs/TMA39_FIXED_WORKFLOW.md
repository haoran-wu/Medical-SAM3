# Fixed TMA39-to-SAM3 Workflow

`config/tma39_fixed_workflow_v1.json` is the source of truth. This is the only accepted pre-VLM candidate workflow for TMA39 and for transfer to another TMA.

## Fixed pipeline

1. Use the registered H&E and the K=12 strict-official raw FICTURE image. Do not smooth, recolor, or reinterpret its 12-factor RGB values.
2. Use the frozen TMA39 set of 120 genes and nine GeneMap modules. For another TMA, keep the same gene-to-module membership and calculate new spatial scores from that TMA's transcripts.
3. In each module, use the top 1% score bins as centers, top 2.5% bins to confirm connection, and connected top 5% bins to define region extent. Merge same-module regions only when they share at least three edge-adjacent GeneMap bins.
4. Place one tight box with 0% expansion around each final GeneMap region and one positive point at its highest module score.
5. Send every prompt to SAM3-base once on raw FICTURE and once on registered H&E. Require the selected mask to contain the point. Select the mask by GeneMap-support F1. Keep every selected FICTURE mask as a primary candidate.
6. Keep a same-prompt H&E mask only if at least 40% of its area lies outside the union of all FICTURE primary masks.
7. Independently detect H&E-present/FICTURE-low regions with fixed 64-pixel cells, H&E tissue at least 35%, FICTURE signal at most 8%, and at least six connected cells. Use one 10%-context box plus one automatic inside point. Keep the highest-SAM-score unique mask containing the point.
8. Deduplicate the full pool at IoU 0.90. Do not union masks.
9. Give the VLM six registered views per frozen candidate: close, medium, and full H&E plus the same three raw-FICTURE views. Use a thin neutral boundary and include the fixed RGB legend in the text prompt.
10. Use official annotation only for posthoc evaluation. A candidate is class-evaluable when at least 50% of its area lies inside one official region.
11. The class-grouped second SAM is planned, not completed: accepted candidates with the same VLM class will be sent together as boxes and inside points.

Prompt counts and retained H&E counts are data-dependent. TMA39 produced 44 main prompts and 55 final candidates. Applying the same rules to TMA30 produced 45 main prompts and 52 final candidates; the different counts do not represent a parameter change.

## Verified TMA30 transfer

- GeneMap: 120 active genes, nine frozen modules, 45 final regions.
- First SAM3: 45 FICTURE masks plus 45 paired H&E masks; one call per prompt per source.
- Supplements: three same-prompt H&E masks and four independent H&E masks.
- Frozen pool: 52 unique masks after IoU 0.90 deduplication; no masks were unioned.
- VLM: `google/gemma-4-31b-it`, 52/52 candidates, six image hashes per candidate.
- Posthoc evaluation: 20 correct among 22 evaluable candidates, or 90.9%.
- Final report SHA-256: `316d420127e5dbd8da974709e6275bdab688e77ada64eace581bd02062e807b4`.

Run `python scripts/validate_tma39_fixed_workflow.py --run-root <RUN_ROOT>` before presenting or packaging a transferred run.

## Rejected legacy route

The deleted TMA30 route used 1280-by-1280 sliding boxes with stride 640, did not build GeneMaps, and created a 51-candidate H&E/FICTURE grid pool. It is not a valid transfer of this workflow and must not be restored or cited as the current method.
