# graph-training
Generation of training data for the graph extraction of endoscopic images.

## Workflow
Functions | Description
--- | ---
``` before_filter.py``` | Extracts and crops video frames
```filtering/Bladder_vessels.m``` | Applies B-COSFIRE filter to cropped images


## Folders
Folder | Description
---| ---
`raw` | Raw video stills
`cropped` | Cropped images
`filtered` | Filtered images