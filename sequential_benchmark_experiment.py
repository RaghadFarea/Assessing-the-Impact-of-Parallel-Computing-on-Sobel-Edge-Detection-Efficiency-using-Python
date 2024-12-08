import cv2
import os
import numpy as np
import math
import time


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

    x = 0
    images = []
    for root, dircs, files in os.walk(folder_pth):
        for file in files:
            if file.endswith(('.png', '.jpg', '.bmp')):
                image_path = os.path.join(root, file)
                imag = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if imag is not None:
                    images.append(imag)
                else:
                    print(f"Image {image_path} not found!")

    return images

#######################################################################################


# the comments inside the sobel_algo function represent benchmarks for diffreint levels of For loops
def sobel_algo(images):

    # total_images = 0
    print("\n START OF THE SOBEL")
    results = []
    start_time = time.time()
    for img in images:

        result = np.zeros_like(img, dtype=np.uint8)
        print(1)

        # start_time = time.time()
        for y in range(1, img.shape[0] - 1):
            for x in range(1, img.shape[1] - 1):
                # sum_of_pixels = 0
                # count = 0
                gx = 0
                gy = 0
                # start_time = time.perf_counter()
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        pixel = img[y + k, x + l]
                        gx += pixel * Gx[k + 1, l + 1]
                        gy += pixel * Gy[k + 1, l + 1]

                # to calculate the of the pixl magnitude
                magnitude = int(math.sqrt(gx * gx + gy * gy))
                magnitude = max(0, min(255, magnitude))
                result[y, x] = magnitude
                # end_time = time.perf_counter()
                # elapsed_time = end_time - start_time
                # sum_of_pixels += elapsed_time
                # count += 1
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # total_images += elapsed_time
        # total_pixels += 1

        results.append(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    # avg_time = total_images / 10
    # print("\n", avg_time)
    # avg_processing_time = sum_of_pixels / count
    # avg_processing_time_us = avg_processing_time * 1_000_000
    # print(
    #     f"\nAverage processing time for image: {avg_processing_time_us:.6f} ms")
    # print("\n",)
    print("\n END OF THE SOBEL")
    return results


#######################################################################################


def process_images(folder_path, main_path):

    folder_path = os.path.join(main_path, folder_path)
    if not os.path.isdir(folder_path):
        print(
            f"Folder '{folder_path}' does not exist in the base directory '{folder_path}'.")
        return

    images = read_images(folder_path)

    sobel_images = sobel_algo(images)

    new_folder = ""

    save_images(sobel_images, main_path)


#######################################################################################

def save_images(sobel_images, folder_path):
    """
    saving processed imges to new folder
    """
    output_folder = os.path.join(folder_path, 'output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, sobel in enumerate(sobel_images):
        sobel_output_path = os.path.join(
            output_folder, f"sobel_image_{idx + 1}.jpg")
        cv2.imwrite(sobel_output_path, sobel)

        print(f"Saved Sobel image to: {sobel_output_path}")

#######################################################################################


def main():
    # change the folder paths to test diffreint types of images
    folder_path = r"c:\Users\Hp\Desktop\sobel algorithm code\dataset"
    main_path = r"c:\Users\Hp\Desktop\sobel algorithm code"
    process_images(folder_path, main_path)
    print("\nEND")

#######################################################################################


if __name__ == "__main__":
    main()
