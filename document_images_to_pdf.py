#!/usr/bin/env python3
import skimage.io as skiio
import numpy as np
import sys
import os
import cv2
import skimage.morphology as skimo
import skimage.transform as skit
import skimage.exposure as skie
import skimage.color as skic
import scipy.ndimage as scipynd
import scipy.interpolate as scipyi
import warnings
import argparse
import fpdf
import matplotlib.pyplot as plt
import math


parser = argparse.ArgumentParser(description="clean document-photographs")

parser.add_argument('files', metavar="INPUT_IMAGES", type=str, nargs='+', help="input-images to clean")
#parser.add_argument('-t','--threshold', metavar="THRESHOLD", type=float, default=0.80, help="grayscale-threshold used to find corners in binariesed image")
parser.add_argument('-g','--grayscale', help="converts image to grayscale", action="store_true")
parser.add_argument('-s','--size', metavar="OUTPUTWIDTH", type=int, default=2000, help="specify output-width of clean image")
parser.add_argument('-e','--enhance_text', help="enhance text using an point-operation to transform colours", action="store_true")


sqrt2 = 1.4142135623730951454746218587388284504413604736328125
processing_width = 500
threshold = "comb"



def gauss(x, sigma=1):
    return (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-math.pow(x,2)/(2*math.pow(sigma,2)))
gauss_vectorised = np.vectorize(gauss)
def gauss_v_normalised(xarr, sigma=1):
    g_arr = gauss_vectorised(xarr, sigma)
    return g_arr/g_arr.sum()
def gauss_kernel(size, sigma=1):
    return gauss_v_normalised(np.arange(size)-int(size/2), sigma)

def get_transf_func(thresh):
    points = [[0,0], [thresh,0], [thresh,thresh], [thresh,255], [255,255]]
    x = np.array(points)[:,0]
    y = np.array(points)[:,1]

    num_points = len(points)
    ipl_t = np.linspace(0.0, 1, num_points-2, endpoint=True)
    ipl_t = np.append([0,0,0],ipl_t)
    ipl_t = np.append(ipl_t,[1,1,1])

    tck = [ipl_t, [x,y], 3]
    u3 = np.linspace(0,1,2000, endpoint=True)

    interpolation = scipyi.splev(u3,tck)

    x_i = np.round(interpolation[0]).astype(np.uint8)
    y_i = list()
    for i in range(256):
        indices = np.where(x_i == i)
        num_indices = np.array(indices).shape[-1]
        i_sum = np.sum(interpolation[1][indices])
        y_i.append(i_sum/num_indices)
    y_i = np.round(np.array(y_i)).astype(np.uint8)
    return y_i

def get_paper_thresh(hist):
    b2 = np.array(list(hist)+[0], dtype=np.float)
    b1 = np.array([0]+list(hist), dtype=np.float)
    d_hist = (b2-b1)/2
    d_hist = d_hist[1:]

    filter_kernel_gauss = gauss_kernel(21,5)
    margin = int(len(filter_kernel_gauss)/2)
    d_hist = np.convolve(d_hist, filter_kernel_gauss)[margin:-margin]
    d_hist = np.convolve(d_hist, filter_kernel_gauss)[margin:-margin]
    d_hist = np.convolve(d_hist, filter_kernel_gauss)[margin:-margin]

    argmax_d_hist = np.argmax(d_hist)
    max_d_hist = d_hist[argmax_d_hist]
    white_d_hist_threshold = int(max_d_hist/100)
    args_d_hist_low = np.where(d_hist<=white_d_hist_threshold)
    args_d_hist_low = np.where(args_d_hist_low<argmax_d_hist, args_d_hist_low, 0)
    paper_thresh = args_d_hist_low.max()

    return paper_thresh


def get_otsu_thresh(hist):
    hist_sum = hist.sum()
    hist_mean = hist.mean()

    q_list = []
    for t in list(range(len(hist)))[1:]:
        h0 = np.zeros_like(hist)
        h1 = np.zeros_like(hist)
        h0[:t] = hist[:t]
        h1[t:] = hist[t:]
        hist_mean0 = h0[:t].mean()
        hist_mean1 = h1[t:].mean()
        hist_sum0 = h0.sum()
        hist_sum1 = h1.sum()

        sigma0 = np.sum(np.square(h0-hist_mean0)*(h0/hist_sum))
        sigma1 = np.sum(np.square(h1-hist_mean0)*(h1/hist_sum))

        p0 = h0.sum()/hist_sum
        p1 = h1.sum()/hist_sum

        sigma_in = p0*sigma0 + p1*sigma1

        sigma_zw = p0*np.square(hist_mean0-hist_mean) + p1*np.square(hist_mean1-hist_mean)

        q = sigma_zw/sigma_in
        q_list.append(q)
    q_list = np.array(q_list)
    otsu_thresh = np.argmax(q_list)
    return otsu_thresh


def text_enhancing_point_transform(input_image):

    # get threshold for white pixels
    hist,bins = np.histogram(input_image, bins=np.arange(256))
    bins = bins[1:]

    if threshold == "paper":
        paper_thresh = get_paper_thresh(hist)
        y_i = get_transf_func(paper_thresh)
    elif threshold == "otsu":
        otsu_thresh = get_otsu_thresh(hist)
        y_i = get_transf_func(otsu_thresh)
    elif threshold == "comb":
        paper_thresh = get_paper_thresh(hist)
        otsu_thresh = get_otsu_thresh(hist)
        comb_thresh = int(0.4*otsu_thresh+0.6*paper_thresh)
        y_i = get_transf_func(comb_thresh)

    return y_i[input_image]


def find_corners(binary_image):
    min_val = binary_image.min()
    max_val = binary_image.max()
    img_thresh = min_val+(0.8*(max_val-min_val))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binary_image = skie.equalize_adapthist(binary_image, kernel_size=processing_width/50)
    binary_image = np.where(binary_image>img_thresh,1,0)

    binary_image = skimo.binary_erosion(binary_image, skimo.disk(processing_width/200))
    binary_image = skimo.binary_dilation(binary_image, skimo.disk(processing_width/200))

    sx = scipynd.sobel(binary_image, axis=0, mode="constant")
    sy = scipynd.sobel(binary_image, axis=1, mode="constant")
    binary_image_edges = np.hypot(sx,sy)
    binary_image_edges = np.where(binary_image_edges > 0.0, 1, 0)

    h,w = binary_image.shape
    tri_size_w = int(w/2)
    tri_size_h = int(h/2)

    ci_ur = np.triu_indices(w,tri_size_w)
    ci_ul = [ci_ur[0],w-1-ci_ur[1]]
    ci_ll = np.tril_indices(h,-(h-tri_size_w))
    ci_lr = [ci_ll[0],w-1-ci_ll[1]]

    corner_mask = np.zeros_like(binary_image_edges)
    corner_mask[ci_ul] = 1
    binary_image_ul = binary_image_edges*corner_mask
    white_points_ul = np.where(binary_image_ul==1)
    white_positions_ul = np.column_stack([white_points_ul[1],white_points_ul[0]])
    ul_positions = np.array(white_positions_ul)+1
    ul = np.sum(ul_positions**2, axis=1)
    min_ul = np.argmin(ul)
    topl = ul_positions[min_ul]-1

    corner_mask[:] = 0
    corner_mask[ci_ur] = 1
    binary_image_ur = binary_image_edges*corner_mask
    white_points_ur = np.where(binary_image_ur==1)
    white_positions_ur = np.column_stack([white_points_ur[1],white_points_ur[0]])
    ur_positions = np.array(white_positions_ur)+1
    ur_positions = np.array(white_positions_ur)
    ur_positions[:,0] = w - ur_positions[:,0]
    ur_positions = ur_positions+1
    ur = np.sum(ur_positions**2, axis=1)
    min_ur = np.argmin(ur)
    topr = ur_positions[min_ur]-1
    topr[0] = w - topr[0]

    corner_mask[:] = 0
    corner_mask[ci_ll] = 1
    binary_image_ll = binary_image_edges*corner_mask
    white_points_ll = np.where(binary_image_ll==1)
    white_positions_ll = np.column_stack([white_points_ll[1],white_points_ll[0]])
    ll_positions = np.array(white_positions_ll)+1
    ll_positions = np.array(white_positions_ll)
    ll_positions[:,1] = h - ll_positions[:,1]
    ll_positions = ll_positions+1
    ll = np.sum(ll_positions**2, axis=1)
    min_ll = np.argmin(ll)
    botl = ll_positions[min_ll]-1
    botl[1] = h - botl[1]

    corner_mask[:] = 0
    corner_mask[ci_lr] = 1
    binary_image_lr = binary_image_edges*corner_mask
    white_points_lr = np.where(binary_image_lr==1)
    white_positions_lr = np.column_stack([white_points_lr[1],white_points_lr[0]])
    lr_positions = np.array(white_positions_lr)+1
    lr_positions = np.array(white_positions_lr)
    lr_positions[:,0] = w - lr_positions[:,0]
    lr_positions[:,1] = h - lr_positions[:,1]
    lr_positions = lr_positions+1
    lr = np.sum(lr_positions**2, axis=1)
    min_lr = np.argmin(lr)
    botr = lr_positions[min_lr]-1
    botr[0] = w - botr[0]
    botr[1] = h - botr[1]


    return topl,topr,botl,botr

if __name__ == "__main__":
    args = parser.parse_args()

    a4_width = args.size
    grayscale = args.grayscale
    enhance_text = args.enhance_text
#    threshold = args.threshold
    images_list = args.files

    a4_size = (a4_width,int(a4_width*sqrt2+0.5))
    wa4,ha4 = a4_size

    p_size = (processing_width, int(processing_width*sqrt2+0.5))
    p_size_w,p_size_h = p_size


    endings = [".jpg", ".png", ".gif"]
    images_list = [i for i in images_list if any(e in i for e in endings)]
    images_list = sorted(images_list)

    for filename in images_list:
        print(filename, end=" - ", flush=True)
        original_image = np.array(skiio.imread(filename), dtype=np.uint8)
        h,w = original_image.shape[:2]
        h_factor = h/p_size_h
        w_factor = w/p_size_w

        print("find_corners", end=" - ", flush=True)
        binary_image = np.array(original_image, dtype=np.uint8)
        binary_image = skit.resize(binary_image, (p_size_h,p_size_w), mode="constant")
        if len(binary_image) >2:
            binary_image = skic.rgb2gray(binary_image)

        topl,topr,botl,botr = find_corners(binary_image)
        topl = (topl[0]*w_factor, topl[1]*h_factor)
        topr = (topr[0]*w_factor, topr[1]*h_factor)
        botl = (botl[0]*w_factor, botl[1]*h_factor)
        botr = (botr[0]*w_factor, botr[1]*h_factor)

        print("transform", end=" - ", flush=True)
        points_is = np.array([topl,topr,botl,botr], dtype=np.float32)
        points_shall = np.array([[0,0], [wa4,0], [0,ha4], [wa4,ha4]], dtype=np.float32)

        warp_mat = cv2.getPerspectiveTransform(points_is, points_shall)

        image = cv2.warpPerspective(original_image, warp_mat, a4_size, flags=cv2.INTER_NEAREST)

        if grayscale or enhance_text:
            print("to_grayscale", end=" - ", flush=True)
            if len(image.shape) > 2:
                image = np.round(skic.rgb2gray(image)*255).astype(np.uint8)

        print("extend_contrast", end=" - ", flush=True)
        image = np.array(image, dtype=np.float)
        valmin,valmax = image.min(),image.max()
        image = np.round((image-valmin)/(valmax-valmin)*255).astype(np.uint8)

        if enhance_text:
            print("enhance_text", end=" - ", flush=True)
            image = 255-np.array(cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,np.ones((20,20))), dtype=np.uint8)
            image = text_enhancing_point_transform(image).astype(np.uint8)
            image = (skie.equalize_adapthist(image)*255).astype(np.uint8)


        save_head,save_tail = os.path.split(filename)
        save_tail = "clean_{}".format(save_tail)
        save_filename = os.path.join(save_head,save_tail)
        print("save {}".format(save_filename))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skiio.imsave(save_filename, image)

    pdf = fpdf.FPDF(orientation="P", unit="pt", format="A4")

    a4_size = [a4w, a4h] = [pdf.fw_pt, pdf.fh_pt]
    image_size = [image_w, image_h] = [a4w*0.95, a4h*0.95]
    margin_size = [margin_w, margin_h] = [(a4w*0.05)/2, (a4h*0.05)/2]

    for filename in images_list:
        fhead,ftail = os.path.split(filename)
        ftail = "clean_{}".format(ftail)
        fname = os.path.join(fhead,ftail)

        pdf.add_page()
        pdf.image(fname, margin_w, margin_h, image_w, image_h)

    pdf.output("document_images_to_pdf.pdf", "F")

    for filename in images_list:
        fhead,ftail = os.path.split(filename)
        ftail = "clean_{}".format(ftail)
        fname = os.path.join(fhead,ftail)
        os.remove(fname)
