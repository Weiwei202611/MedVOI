# Dataset manifest specification

MedVOI is intentionally manifest-driven. The repository does not download or redistribute medical datasets.

## BraTS 2021

Required CSV columns:

```text
patient_id,split,t1,t1ce,t2,flair,label
```

`split` must be one of `train`, `val`, `test`. The manuscript uses 876/125/250 patient-level cases.

The label can be an integer label map or a 3-channel foreground mask. The reference implementation interprets integer labels 1..C as C foreground channels.

## PI-CAI

```text
patient_id,split,t2w,dwi,adc,label
```

`label` is a binary clinically significant prostate cancer target. Use a fixed patient-level stratified 70/10/20 split if reproducing the manuscript design.

## Symile-MIMIC

```text
patient_id,split,cxr,ecg,labs,label
```

- `cxr`: path to a single-channel 2D tensor/array
- `ecg`: path to a 12 x T tensor/array
- `labs`: path to a length-50 tensor/array
- `label`: in-hospital mortality (0/1)

Use the official released query splits and do not re-randomize patient membership.

## autoPET

```text
patient_id,split,pet,ct,label
```

`label` is the lesion segmentation mask. Multiple studies from one patient must remain in the same split.

## Supported file formats

The generic loader supports:

- `.npy`
- `.npz`
- `.pt` / `.pth`
- `.nii` / `.nii.gz`

For large-scale production runs, users may replace the generic loader with a cached or streaming loader while keeping the same model interfaces.
