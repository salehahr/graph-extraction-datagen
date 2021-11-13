import cv2
import glob

if __name__ == '__main__':
    cropped_files = glob.glob('./**/cropped/*.png', recursive=True)
    filtered_files = glob.glob('./**/filtered/*.png', recursive=True)
    masked_files = glob.glob('./**/masked/*.png', recursive=True)
    threshed_files = glob.glob('./**/threshed/*.png', recursive=True)

    files_to_resize = cropped_files + filtered_files + masked_files + threshed_files
    # print(files_to_resize)

    new_length = 256

    for fp in files_to_resize:
        img = cv2.imread(fp)
        cv2.resize(img, (new_length, new_length))
        cv2.imwrite(fp, img)