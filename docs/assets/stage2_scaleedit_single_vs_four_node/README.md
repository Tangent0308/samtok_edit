# ScaleEdit single-node versus four-node-trained comparison

This directory stores the checked-in category overviews for E21 in
`SamtokEdit_实验记录.md`. The evaluation uses the same 32-sample ScaleEdit
validation metadata and identical inference settings for every generated
setting.

Each edit panel contains seven columns:

1. source image;
2. ground-truth edited image;
3. stock Qwen-Image-Edit-2511;
4. single-node/8-GPU-trained online CoT;
5. single-node/8-GPU-trained `edit_umt`;
6. four-node/32-GPU-trained online CoT;
7. four-node/32-GPU-trained `edit_umt`.

Each mask panel contains four separately overlaid columns: raw GT mask, GT
mask-token decode, single-node/8-GPU-trained online decode, and
four-node/32-GPU-trained online decode.

The repository keeps all eight category overviews, all 64 full-resolution
per-case panels, and the two JSONL visualization manifests. The complete
machine-readable metric report and raw inference outputs remain under the
experiment root:

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/
stage2_evaluation/scaleedit_precision_32/single_vs_four_node/
```

## Overview file integrity

```text
289b76ccf34e57e02221ccc2a4055620c38b0748a15658aa6f4f2e26be3d7d3e  edit_comparisons/fine_grained.jpg
a4e450e3f15afbfd44726756061ae9caa1783851b75b55314e576b9e34598f1c  edit_comparisons/multi_instance.jpg
1ce829a805f3d7780ac1ab5dcc2b372c2f4326d711c26bc46b50f85254c04150  edit_comparisons/precise_edit.jpg
162db4b0e977d98173f8ede273ba380435fea64216dcb97682282d1564de5f51  edit_comparisons/small_object.jpg
d68be0ef557ae78c10ad1c50ef53e45653f74de5a7f62be1769ebe6452bf0367  mask_comparisons/fine_grained.jpg
dba0444f3bb9863d3b559492c421026332d685b9e06b0b59c6cc82f644f5e52b  mask_comparisons/multi_instance.jpg
d0e8c3dbc69ab7c5c270687be26cb29145ac8557b781790ba3db3dfd76db9a49  mask_comparisons/precise_edit.jpg
33be300b038feb2dbf75bf86af815da8beacb062f45e0e6002137b13b9803138  mask_comparisons/small_object.jpg
```
