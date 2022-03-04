import json
from matplotlib import pyplot as plt
from PIL import Image
from skimage.exposure import histogram
from skimage.io import imsave, imshow, show, imread
import numpy as np


# save_current_parameters_to_file('D:\\Рабочий стол\\Lab_1_settings.json')
settings = {
    'filepath': 'D:\\Рабочий стол\\TestImages\\01_apc.tif',
    'task_1_parameter': 100,
    'path_to_save_result': "D:\\Рабочий стол\\Images"
}

MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0


def open_image_as_arrays(filepath):
    img_as_arrays = imread(filepath)
    return img_as_arrays


def process_parameters_for_contrasting(f_min, f_max):
    proc_a = MAX_BRIGHTNESS_VALUE/(f_max - f_min)
    proc_b = -MAX_BRIGHTNESS_VALUE*f_min/(f_max - f_min)
    return proc_a, proc_b


def get_max_and_min_brightness_value(img_as_arrays):
    min_el = np.min(img_as_arrays)
    max_el = np.max(img_as_arrays)
    return max_el, min_el


def transformation_function_task_2(value_to_proc):
    elem_value = int(a * value_to_proc + b)
    if elem_value <= MAX_BRIGHTNESS_VALUE:
        return elem_value
    else:
        return MAX_BRIGHTNESS_VALUE


def contrasting(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(transformation_function_task_2, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def transformation_function_task_1(element_value):
    if element_value < TASK_1_PARAMETER:
        return MIN_BRIGHTNESS_VALUE
    else:
        return MAX_BRIGHTNESS_VALUE


def border_processing(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(transformation_function_task_1, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def read_parameters(filepath):
    json_data = json.load(open(filepath))
    for entry in json_data.keys():
        print(f'Red parameter " {entry}" with value = "{json_data[entry]}" ')
    return json_data


def create_figure_of_union_plot(image1, image2, task_function):
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
    create_border_function_graphic(task_function)
    return fig


def create_border_function_graphic(function):
    plt.axis([0, 255, 0, 270])
    plt.title('Transformation function graphic')
    arr_x_values = np.arange(255)
    dots = list(map(function, np.arange(255)))
    plt.plot(arr_x_values, np.array(dots), color='blue', linewidth=2.0)


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
TASK_1_PARAMETER = json_settings_file['task_1_parameter']
path_to_save = json_settings_file['path_to_save_result']

opened_image = open_image_as_arrays(image_filepath)
max_value, min_value = get_max_and_min_brightness_value(opened_image)


create_figure_of_union_plot(opened_image, border_processing(opened_image), transformation_function_task_1)
plt.tight_layout()
show()

print(border_processing(opened_image))
print("For task2 parameters")
print(min_value)
print(max_value)
a, b = process_parameters_for_contrasting(min_value, max_value)
print(a)
print(b)

create_figure_of_union_plot(opened_image, contrasting(opened_image), transformation_function_task_2)
plt.tight_layout()
show()






