import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision

from PIL import ImageEnhance


###
def random_swap_pair(pre_img, post_img, p=0.5):
    if random.random() < p:
        return post_img, pre_img
    return pre_img, post_img

def random_hflip(pre_img, post_img, label, p=0.5):
    if random.random() < p:
        pre_img  = np.fliplr(pre_img).copy()
        post_img = np.fliplr(post_img).copy()
        label    = np.fliplr(label).copy()
    return pre_img, post_img, label

def random_vflip(pre_img, post_img, label, p=0.5):
    if random.random() < p:
        pre_img  = np.flipud(pre_img).copy()
        post_img = np.flipud(post_img).copy()
        label    = np.flipud(label).copy()
    return pre_img, post_img, label

def random_rot90(pre_img, post_img, label, p=0.5):
    if random.random() < p:
        k = random.choice([1, 2, 3])  # 90/180/270
        pre_img  = np.rot90(pre_img,  k).copy()
        post_img = np.rot90(post_img, k).copy()
        label    = np.rot90(label,    k).copy()
    return pre_img, post_img, label

def _crop_box_from_scale_ratio(h, w, scale_range=(0.5, 1.0), ratio_range=(3/4, 4/3)):
    area = h * w
    target_area = random.uniform(*scale_range) * area
    log_ratio   = (np.log(ratio_range[0]), np.log(ratio_range[1]))
    aspect      = np.exp(random.uniform(*log_ratio))
    new_h = int(round(np.sqrt(target_area / aspect)))
    new_w = int(round(np.sqrt(target_area * aspect)))
    if 0 < new_h <= h and 0 < new_w <= w:
        top  = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        return top, left, new_h, new_w
    in_short = min(h, w)
    top  = (h - in_short) // 2
    left = (w - in_short) // 2
    return top, left, in_short, in_short

def random_resized_crop_256_pos(pre_img, post_img, label, p=0.5, out_size=256,
                                scale=(0.5, 1.0), ratio=(3/4, 4/3), tries=30):

    if random.random() >= p:
        return pre_img, post_img, label

    h, w = label.shape
    box = None
    for _ in range(tries):
        top, left, new_h, new_w = _crop_box_from_scale_ratio(h, w, scale, ratio)
        lbl_crop = label[top:top+new_h, left:left+new_w]
        if (lbl_crop > 0.5).any():
            box = (top, left, new_h, new_w)
            break
    if box is None:
        return pre_img, post_img, label

    top, left, new_h, new_w = box
    pre_crop  = pre_img[top:top+new_h,  left:left+new_w,  :]
    post_crop = post_img[top:top+new_h, left:left+new_w,  :]
    lab_crop  = label[top:top+new_h,    left:left+new_w]

    pre_pil  = Image.fromarray(pre_crop.astype(np.uint8))
    post_pil = Image.fromarray(post_crop.astype(np.uint8))
    lab_pil  = Image.fromarray(lab_crop.astype(np.uint8))

    pre_pil  = pre_pil.resize((out_size, out_size), Image.BILINEAR)
    post_pil = post_pil.resize((out_size, out_size), Image.BILINEAR)
    lab_pil  = lab_pil.resize((out_size, out_size), Image.NEAREST)

    pre_out  = np.array(pre_pil,  dtype=np.float32)
    post_out = np.array(post_pil, dtype=np.float32)
    lab_out  = np.array(lab_pil,  dtype=np.float32)

    return pre_out, post_out, lab_out

####



def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()

    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2


def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pad_image

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_image[H_start:H_end, W_start:W_end, 0]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = object.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    H_start = random.randrange(0, H - crop_size + 1, 1)
    H_end = H_start + crop_size
    W_start = random.randrange(0, W - crop_size + 1, 1)
    W_end = W_start + crop_size

    # H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    # post_img = post_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(GT))
    return pre_img, object


##################################################
def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0,0,0], ignore_index=255):
    h, w = label.shape
    H = max(crop_size, h); W = max(crop_size, w)
    pad_pre = np.zeros((H, W, 3), np.float32)
    pad_post = np.zeros((H, W, 3), np.float32)
    pad_lab = np.ones((H, W), np.float32) * ignore_index
    pad_pre[...,0]=mean_rgb[0]; pad_pre[...,1]=mean_rgb[1]; pad_pre[...,2]=mean_rgb[2]
    pad_post[...,0]=mean_rgb[0]; pad_post[...,1]=mean_rgb[1]; pad_post[...,2]=mean_rgb[2]

    Hp = int(np.random.randint(H - h + 1)); Wp = int(np.random.randint(W - w + 1))
    pad_pre[Hp:Hp+h, Wp:Wp+w] = pre_img
    pad_post[Hp:Hp+h, Wp:Wp+w] = post_img
    pad_lab[Hp:Hp+h, Wp:Wp+w] = label

    def pick_box(require_pos=True, tries=50, cat_max_ratio=0.75):
        last = (0, crop_size, 0, crop_size)
        for _ in range(tries):
            hs = random.randrange(0, H - crop_size + 1); ws = random.randrange(0, W - crop_size + 1)
            he, we = hs + crop_size, ws + crop_size
            temp = pad_lab[hs:he, ws:we]
            valid = temp[temp != ignore_index]
            if require_pos and valid.size > 0 and (valid > 0.5).any():
                return hs, he, ws, we
            # 退而求其次：避免单类主导（修正 len(cnt)>1 的判断）
            idx, cnt = np.unique(valid, return_counts=True)
            if len(cnt) > 1 and (np.max(cnt)/np.sum(cnt) < cat_max_ratio):
                return hs, he, ws, we
            last = (hs, he, ws, we)
        return last

    hs, he, ws, we = pick_box(require_pos=True)
    return pad_pre[hs:he, ws:we], pad_post[hs:he, ws:we], pad_lab[hs:he, ws:we]
##################################################


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_2 = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2



