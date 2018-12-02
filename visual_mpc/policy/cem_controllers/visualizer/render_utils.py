import numpy as np
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from visual_mpc.utils.im_utils import npy_to_gif


def add_crosshairs(images, pos, color=None):
    """
    :param images: shape: batch, t, r, c, 3 or list of lenght t with batch, r, c, 3
    :param pos: shape: batch, t, 2
    :param color: color needs to be vector with in [0,1]
    :return: images in the same shape as input
    """
    assert len(pos.shape) == 3

    if isinstance(images, list):
        make_list_output = True
        images = np.stack(images, axis=1)
    else:
        make_list_output = False

    assert len(images.shape) == 5
    assert images.shape[0] == pos.shape[0]


    pos = np.clip(pos, np.zeros(2).reshape((1,1,2)),np.array(images.shape[2:4]).reshape((1,1,2)) -1)

    if color == None:
        if images.dtype == np.float32:
            color = np.array([0., 1., 1.], np.float32)
        else:
            color = np.array([0, 255, 255], np.uint8)

    out = np.zeros_like(images)
    for b in range(pos.shape[0]):
        for t in range(images.shape[1]):
            im = np.squeeze(images[b,t])
            p = pos[b,t].astype(np.int)
            im[p[0]-5:p[0]-2,p[1]] = color
            im[p[0]+3:p[0]+6, p[1]] = color

            im[p[0],p[1]-5:p[1]-2] = color

            im[p[0], p[1]+3:p[1]+6] = color

            im[p[0], p[1]] = color
            out[b, t] = im

    if make_list_output:
        out = np.split(out, images.shape[1], axis=1)
        out = [np.squeeze(el) for el in out]
    return out


def resize_image(input, size = (256, 256)):
    """
    :param input:  list of image batches of size [b, r, c, ch], or [b,t,n,r,c,ch]
    :param size:
    :param mode:
    :return:
    """

    assert len(size) == 2
    out = []
    if isinstance(input, list):
        for im in input:
            if len(im.shape) == 4:
                batch_size, height, width, ch = im.shape
            else:
                batch_size, height, width = im.shape
                ch = 1
                im = im[..., None]

            im = np.transpose(im, [1,2,0,3])
            im = im.reshape(height, width, -1)
            out_t = cv2.resize(im, (size[1], size[0]))
            out_t = out_t.reshape(size[0], size[1], batch_size, ch)
            out_t = np.transpose(out_t, [2, 0, 1, 3])
            out.append(out_t)
    else:
        batch_size, seqlen, ncam, height, width, ch = input.shape

        im = np.transpose(input, [3,4,0,1,2,5])
        im = im.reshape(height, width, -1)
        im = (im*255).astype(np.uint8)
        # out_t = cv2.resize(im, (size[1], size[0]))
        out_t = resize(im, (size[0], size[1]))
        out_t = out_t.reshape(size[0], size[1], batch_size, seqlen, ncam, ch)
        out = np.transpose(out_t, [2,3,4,0,1,5])

        # out = out.astype(np.float32)/255.
    return out


def color_code_distrib(distrib_list, num_ex, renormalize=False):
    # self.logger.log('renormalizing heatmaps: ', renormalize)
    out_distrib = []
    for distrib in distrib_list:
        out_t = []

        for b in range(num_ex):
            cmap = plt.cm.get_cmap('jet')
            if renormalize:
                distrib[b] /= (np.max(distrib[b])+1e-6)
            colored_distrib = cmap(np.squeeze(distrib[b]))[:, :, :3]
            out_t.append(colored_distrib)

            # plt.imshow(np.squeeze(distrib[b]))
            # plt.show()

        out_t = np.stack(out_t, 0)
        out_distrib.append(out_t)

    return out_distrib


def draw_text_image(text, background_color=(255,255,255), image_size=(30, 64), dtype=np.float32):

    from PIL import Image, ImageDraw
    text_image = Image.new('RGB', image_size[::-1], background_color)
    draw = ImageDraw.Draw(text_image)
    if text:
        draw.text((4, 0), text, fill=(0, 0, 0))
    if dtype == np.float32:
        return np.array(text_image).astype(np.float32)/255.
    else:
        return np.array(text_image)


def draw_text_onimage(text, image, color=(255, 0, 0)):
    if image.dtype == np.float32:
        image = (image*255.).astype(np.uint8)
    assert image.dtype == np.uint8
    text_image = Image.fromarray(image)
    draw = ImageDraw.Draw(text_image)
    draw.text((4, 0), text, fill=color)
    return np.array(text_image).astype(np.float32)/255.


def assemble_gif(video_batch, num_exp = 8, convert_from_float = True, only_ind=None):
    """
    :param video_batch: accepts either
        a list of different video batches
        each video batch is a list of [batchsize, 64, 64, 3] with length timesteps, with type float32 and range 0 to 1
        or each element of the list is tuple (video batch, name)
    or
        a list of tuples with (video_batch, name)

    :param only_ind, only assemble this index
    :return:
    """

    if isinstance(video_batch[0], tuple):
        names = [v[1] for v in video_batch]
        video_batch = [v[0] for v in video_batch]
        txt_im = []
        for name in names:
            txt_im.append(draw_text_image(name, image_size=(video_batch[0][0].shape[1], 200)))
        legend_col = np.concatenate(txt_im, 0)
    else:
        legend_col = None
    vid_length = video_batch[0].shape[1]

    #videobatch is a list of [b, t, r, c, 3]

    fullframe_list = []
    for t in range(vid_length):
        if only_ind is not None:
            column_images = [video[only_ind, t] for video in video_batch]
            full_frame = np.concatenate(column_images, axis=0)  # make column
        else:
            column_list = []
            if legend_col is not None:
                column_list.append(legend_col)

            for exp in range(num_exp):
                column_images = []
                for video in video_batch:
                    column_images.append(video[exp, t])
                column_images = np.concatenate(column_images, axis=0)  #make column
                column_list.append(column_images)

            full_frame = np.concatenate(column_list, axis= 1)

        if convert_from_float:
            full_frame = np.uint8(255 * full_frame)

        fullframe_list.append(full_frame)

    return fullframe_list


def get_score_images(scores, height, width, seqlen, numex):
    txt_im = []
    for i in range(numex):
        txt_im.append(draw_text_image(str(scores[i]), image_size=(height, width)))
    textrow = np.stack(txt_im, 0)
    return np.repeat(textrow[:,None], seqlen, axis=1)


def make_direct_vid(dict, numex, gif_savepath, suf):
    """
    :param dict:  dictionary with video tensors of shape bsize, tlen, r, c, 3
    :param numex:
    :param gif_savepath:
    :param suf:
    :param resize:
    :return:
    """
    new_videolist = []
    shapes = []
    for key in dict:
        images = dict[key]
        print('key', key)
        print('shape', images.shape)

        if len(shapes) > 0:   # check that all the same size
            assert images.shape == shapes[-1], 'shape is different!'
        shapes.append(images.shape)
        assert not isinstance(images, list)

        # if 'gen_distrib' in vid[1]:
        #     plt.switch_backend('TkAgg')
        #     plt.imshow(vid[0][0][0])
        #     plt.show()

        if images[0].shape[-1] == 1 or len(images[0].shape) == 3:
            images = color_code_distrib(images, numex, renormalize=True)
        new_videolist.append((images, key))

    framelist = assemble_gif(new_videolist, convert_from_float=True, num_exp=numex)
    framelist.append(np.zeros_like(framelist[0]))
    # save_video_mp4(gif_savepath +'/prediction_at_t{}')
    npy_to_gif(framelist, gif_savepath +'/direct_{}'.format(suf))