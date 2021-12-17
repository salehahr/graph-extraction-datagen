# Video
# VIDEO_FULL_FILEPATH_EXT = 'M:/graph-training/data/GRK008.mp4'
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK008.MP4"
trim_times = [
    [0, 9.6],
    [15, 16.2],
    [26.1, 26.8],
    [28.1, 29.2],
    [29.8, 29.8],
    [32.4, 38.4],
    [39.4, 40.32],
]
is_synthetic = False

video_filepath = "/graphics/scratch/schuelej/sar/data/GRK011.MP4"
trim_times = [[0, 39], [68, 77], [60 + 57, 60 + 59], [2 * 60 + 17, 2 * 60 + 21]]
is_synthetic = False

video_filepath = "/graphics/scratch/schuelej/sar/data/synthetic-bladder6.mp4"
trim_times = None
is_synthetic = True

video_filepath = "/graphics/scratch/schuelej/sar/data/GRK022.MP4"
trim_times = [
    # [4, 10],
    [11, 12],
    [20, 21],
    [22, 24],
    [24.7, 25],
    [26, 27],
    [32, 34],
    [39, 40],
    # [60 + 15, 60 + 19],
    [60 + 33, 60 + 34],
    [60 + 43, 60 + 47],
]
is_synthetic = False

frequency = 25  # Hz
