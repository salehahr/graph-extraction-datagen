# graph-training
Generation of training data for the graph extraction of endoscopic images.

## Workflow
Functions | Description
--- | ---
``` before_filter.py``` | Extracts and crops video frames
```filtering/Bladder_vessels.m``` | Applies B-COSFIRE filter to cropped images
``` after_filter.py``` | Applies: mask, thresholding, skeletonising, graph generation


## Folders
Folder | Description
---| ---
`raw` | Raw video stills
`cropped` | Cropped images
`filtered` | Filtered images
`masked` | Filtered images without vignettes
`threshed` | Thresholded images
`skeleton` | Skeletonised images
`landmarks` | Extracted landmarks on the skeleton
`poly_graph` | Polynomial graph
`overlay` | Graph overlaid on cropped image
