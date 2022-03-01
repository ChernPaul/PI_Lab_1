import json
from matplotlib import pyplot as plt
from PIL import Image
from skimage.exposure import histogram
from skimage.io import imsave, imshow, show, imread
import numpy as np


# save_current_parameters_to_file('D:\\Рабочий стол\\Lab_1_settings.json')
settings = {
    'filepath': 'D:\\Рабочий стол\\TestImages\\01_apc.tif',
    # MIN_VALUE = 0
    'left_contrast_border_value': 20,
    # MAX_VALUE = 100
    'right_contrast_border_value': 60,
    'path_to_save_result': "D:\\Рабочий стол\\Images"
}

MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0
TASK_1_PARAMETER = 100


def open_image(filepath):
    img_as_arrays = imread(filepath)
    return img_as_arrays


def transformation_function_task_1(element_value):
    if element_value < TASK_1_PARAMETER:
        return MIN_BRIGHTNESS_VALUE
    else:
        return MAX_BRIGHTNESS_VALUE


def border_processing(img):
    shape = np.shape(img)
    new_img_list = list(map(transformation_function_task_1, np.reshape(img, img.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def read_parameters(filepath):
    json_data = json.load(open(filepath))
    for entry in json_data.keys():
        print(f'Red parameter " {entry}" with value = "{json_data[entry]}" ')
    return json_data


def create_figure_of_union_plot(image1, image2):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(3, 2, 1)
    plt.title("Source image")
    imshow(image1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(3, 2, 2)
    plt.title("Processed image")
    imshow(image2, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(3, 2, 3)
    plt.title("Source image Histogram")
    create_wb_histogram_plot(image1)
    fig.add_subplot(3, 2, 4)
    plt.title("Processed image Histogram")
    create_wb_histogram_plot(image2)
    fig.add_subplot(3, 2, 5)
    create_border_function_graphic()
    return fig


def create_border_function_graphic():
    plt.axis([0, 255, 0, 260])
    plt.title('Transformation function graphic')
    arr_x_values = np.arange(255)
    dots = list(map(transformation_function_task_1, np.arange(255)))
    plt.plot(arr_x_values, np.array(dots), linewidth=2.0)


def save_current_parameters_to_file(filepath):
    json.dump(settings, open(filepath, 'w'))


def save_image(img, directory):
    imsave(directory, img)


def create_color_histogram_plot(img_array):
    hist_red, bins_red = histogram(img_array[:, :, 2])
    hist_green, bins_green = histogram(img_array[:, :, 1])
    hist_blue, bins_blue = histogram(img_array[:, :, 0])

    plt.ylabel('Number of counts')
    plt.xlabel('Brightness')
    plt.title('Histogram of the brightness distribution for each channel')
    plt.plot(bins_green, hist_green, color='green', linestyle='-', linewidth=1)
    plt.plot(bins_red, hist_red, color='red', linestyle='-', linewidth=1)
    plt.plot(bins_blue, hist_blue, color='blue', linestyle='-', linewidth=1)
    plt.legend(['green', 'red', 'blue'])


def create_wb_histogram_plot(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)








save_current_parameters_to_file('D:\\Рабочий стол\\Lab_1_settings.json')
json_settings_file = json.load(open('D:\\Рабочий стол\\Lab_1_settings.json'))
image_filepath = json_settings_file['filepath']
contrast_left_border = json_settings_file['left_contrast_border_value']
contrast_right_border = json_settings_file['right_contrast_border_value']
path_to_save = json_settings_file['path_to_save_result']
opened_image = open_image(image_filepath)

create_figure_of_union_plot(opened_image, border_processing(opened_image))
plt.tight_layout()
show()

print(border_processing(opened_image))









