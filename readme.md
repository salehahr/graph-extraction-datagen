# graph-training
Generation of training data for the graph extraction of endoscopic images.

## Workflow
### Video Workflow
1. Append new video data to end of [`video_data.py`](video_data.py)
2. Run `before_filter.py`
3. In MATLAB, run `filtering/Bladder_vessels.m` (filepath can be input manually)
4. Run `after_filter.py`  

### Image Folder Workflow
1. In `video_data.py`:
   1. Update/Set variable `video_filepath` = "C:/My/FullPath/ImageFolder" to folder containing the .png images
   2. Set variable `use_images` = True
   3. If you want to use the FFT filter, set variable `fft_filter` = True
2. Run `before_filter.py`
3. In MATLAB, `filtering/Bladder_vessels.m`:
   1. Update/Set variable line 30  `VIDEO_FILEPATH` = 'C:/My/FullPath/ImageFolder'
   2. Run `filtering/Bladder_vessels.m`
4. Run `after_filter.py`  
  
Functions | Description
--- | ---
``` before_filter.py``` | Extracts and crops video frames
```filtering/Bladder_vessels.m``` | Applies B-COSFIRE filter to cropped images
``` after_filter.py``` | Applies: mask, thresholding, skeletonising, graph generation


## Folders
Folder | Description
---| ---
`raw` | Raw video stills
`cropped` | Cropped images, 256x256px
`filtered` | Filtered images
`masked` | Filtered images masked with a circular mask
`threshed` | Thresholded images
`skeleton` | Skeletonised images
`graphs` | Graphs saved as .json files
`overlay` | Graph overlaid on cropped image
