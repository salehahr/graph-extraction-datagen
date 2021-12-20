# note/status: need to check video regions, possibly refilter
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK008.MP4"
trim_times = [
    [0, 9.6],
    [15, 16.2],
    [26.1, 26.8],
    [28.1, 29.2],
    # [29.8, 29.8],
    [32.4, 38.4],
    [39.4, 44],
]
is_synthetic = False

# note/status: need to check video regions, possibly refilter
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK011.MP4"
trim_times = [[0, 39], [68, 77], [60 + 57, 60 + 59], [2 * 60 + 17, 2 * 60 + 21]]
is_synthetic = False

# note/status: done; these settings are for all synthetic recordings
video_filepath = "/graphics/scratch/schuelej/sar/data/synthetic-bladder6.mp4"
trim_times = None
is_synthetic = True

# note/status: need to refilter
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

# note/status: video very jumpy, done but not much data
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK020.mpg"
trim_times = [
    # [0, 1],
    # [3, 5],
    # [6.7, 10],
    # [10.5, 13],
    [16.6, 21],
    # [29, 35],
    # [39, 39.2],
    # [42.5, 43.7],
]
is_synthetic = False

# note/status: done, but not much data
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK016.mpg"
trim_times = [[19, 25]]  # [0, 10],
is_synthetic = False

# note/status: might discard, too dense
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK012-vlc.mp4"
trim_times = [
    [5.16, 18],
    [35, 36],
    [60 + 27, 60 + 33],
]
is_synthetic = False

# note/status: not processed yet because of a bug in extract_graph
# last successful: 0000_26600
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK021.mpg"
trim_times = [
    [12, 19],
    [21, 24],
    [26, 32],
    [38, 47],
    [50, 52],
    [60 + 6, 60 + 9],
    [60 + 40, 60 + 43],
    [60 + 46, 60 + 52],
]
is_synthetic = False

# note/status: not processed yet
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK015.mpg"
trim_times = [
    [1, 4],
    [10, 12],
    [60 + 28, 60 + 30],
    [60 + 41, 60 + 42],
]
is_synthetic = False

# note/status: not processed yet
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK014.mpg"
trim_times = [
    [2 * 60 + 8, 2 * 60 + 9.44005],
    [2 * 60 + 11, 2 * 60 + 16],
    [5 * 60 + 29, 5 * 60 + 36],
    [6 * 60 + 20, 6 * 60 + 25],
    [6 * 60 + 32, 6 * 60 + 33.5],
    [6 * 60 + 42, 6 * 60 + 46],
    [9 * 60 + 7, 9 * 60 + 8],
    [9 * 60 + 11, 9 * 60 + 13],
]
is_synthetic = False

# note/status: not processed yet
video_filepath = "/graphics/scratch/schuelej/sar/data/GRK007.mpg"
trim_times = [
    [0, 2],
    [60 + 3, 60 + 10],
    [60 + 23, 60 + 46],
    [2 * 60 + 54, 2 * 60 + 57],
    [3 * 60 + 36, 3 * 60 + 39],
    [4 * 60 + 16.5, 4 * 60 + 20],
]
is_synthetic = False

frequency = 25  # Hz
