import cv2
import os
import numpy as np
import math
import time
from multiprocessing import Pool

#######################################################################################

# Sobel kernels
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[-1, -2, -1],
               [0,  0,  0],
               [1,  2,  1]])

#######################################################################################

def read_images(folder_pth):
    print("\nthe start of the read image function")
    x = 0
    images = []
    for root, dircs, files in os.walk(folder_pth):
        for file in files:
            if file.endswith(('.png', '.jpg', '.bmp')):
                image_path = os.path.join(root, file)
                imag = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if imag is not None:
                    x += 1
                    print(f"Processing image: {file}")
                    images.append(imag)
                else:
                    print(f"Image {image_path} not found!")
    print(f"\n read image successfully {x} Images")
    return images

#######################################################################################

def sobel_single_image(image):

    image_count = 1
    pixel_count = image.shape[0] * image.shape[1]
    result = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            gx = 0
            gy = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pixel = image[y + k, x + l]
                    gx += pixel * Gx[k + 1, l + 1]
                    gy += pixel * Gy[k + 1, l + 1]

            magnitude = int(math.sqrt(gx * gx + gy * gy))
            magnitude = max(0, min(255, magnitude))
            result[y, x] = magnitude

    return result, image_count, pixel_count
#########################################################################################

def process_images_in_parallel(images):
    print("\nThe start of parallel processing")
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing.")
    start_time = time.time()

    with Pool(num_cores) as pool:
        results  = pool.starmap(sobel_single_image, images)

    sobel_images, image_counts, pixel_counts = zip(*results)
    total_images = sum(image_counts)
    total_pixels = sum(pixel_counts)
    end_time = time.time()

    print("\nThe end of parallel processing")
    print(f"\nNumber of images processed: {total_images}")
    print(f"Total number of pixels processed: {total_pixels}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    return sobel_images

#######################################################################################

def process_images(folder_path, main_path):
    folder_path = os.path.join(main_path, folder_path)
    if not os.path.isdir(folder_path):
        print(
            f"Folder '{folder_path}' does not exist in the base directory '{folder_path}'.")
        return

    images = read_images(folder_path)
    sobel_images = process_images_in_parallel(images)
    save_images(sobel_images, main_path)

#######################################################################################

def save_images(sobel_images, folder_path):
    print("\nthe start of save image function")
    output_folder = os.path.join(folder_path, 'output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, sobel in enumerate(sobel_images):
        sobel_output_path = os.path.join(output_folder, f"sobel_image_{idx + 1}.jpg")
        cv2.imwrite(sobel_output_path, sobel)
        print(f"Saved Sobel image to: {sobel_output_path}")
    print("\nthe end of the save image function")

#######################################################################################

def main():
    main_path = os.getcwd()
    folder_path = os.path.join(main_path, "Dataset\\Outerloop\\Full_Run")

    print('\nstart')
    
    process_images(folder_path, main_path)
    print("\nEND")

#######################################################################################

if __name__ == "__main__":
    main()
