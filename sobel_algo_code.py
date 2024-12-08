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


def sobel_algo(images):

    print("\n START OF THE SOBEL")
    x = 0
    results = []
    for img in images:

        result = np.zeros_like(img, dtype=np.uint8)

        for y in range(1, img.shape[0] - 1):
            for x in range(1, img.shape[1] - 1):

                gx = 0
                gy = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        pixel = img[y + k, x + l]
                        gx += pixel * Gx[k + 1, l + 1]
                        gy += pixel * Gy[k + 1, l + 1]

                magnitude = int(math.sqrt(gx * gx + gy * gy))
                magnitude = max(0, min(255, magnitude))
                result[y, x] = magnitude

        results.append(result)

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

    main_path = os.getcwd()
    folder_path = os.path.join(main_path, "Dataset\\Outerloop\\Full_Run")

    process_images(folder_path, main_path)
    print("\nEND")

#######################################################################################


if __name__ == "__main__":
    main()
