import cv2
import moviepy.editor as mpy
import os


def resize_store(t, target_array, input_array):
    target_img_height, target_img_width = target_array.shape[2:4]

    if (target_img_height, target_img_width) == input_array.shape[1:3]:
        for i in range(input_array.shape[0]):
            target_array[t, i] = input_array[i]
    else:
        for i in range(input_array.shape[0]):
            target_array[t, i] = cv2.resize(input_array[i], (target_img_width, target_img_height),
                                            interpolation=cv2.INTER_AREA)


def npy_to_gif(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def npy_to_mp4(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_videofile(filename + '.mp4')
