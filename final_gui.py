import cv2
import numpy as np
import math

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from tkinter import messagebox
# from tkinter.ttk import *

from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

#from LDA_class import *#

import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

root = Tk()
root.title("RunWay Detection Software")
root.geometry("800x650")


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=8)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def KNB(mw, K):
    mwg = rgb2gray(mw)
    r = np.size(mwg)

    re, co = np.shape(mwg)
    nbh = np.zeros([re, co, 3])
    # nbh_t = zeros([r,3])
    nbh_v = np.zeros([K, 3])
    cent = mwg[int(np.floor(re / 2)), int(np.floor(co / 2))]
    dist = np.zeros(r)
    dist = abs(mwg - cent)
    dist_ord = np.sort(np.ravel(dist[:]))
    dist_K = dist_ord[K - 1]
    x, y = np.where(dist <= dist_K)
    nbh[x, y, 0] = mw[x, y, 0]
    nbh[x, y, 1] = mw[x, y, 1]
    nbh[x, y, 2] = mw[x, y, 2]
    nbh_t0 = sorted(np.ravel(nbh[:, :, 0]), reverse=True)
    nbh_t1 = sorted(np.ravel(nbh[:, :, 1]), reverse=True)
    nbh_t2 = sorted(np.ravel(nbh[:, :, 2]), reverse=True)
    nbh_v[:, 0] = nbh_t0[0:K]
    nbh_v[:, 1] = nbh_t1[0:K]
    nbh_v[:, 2] = nbh_t2[0:K]
    return nbh_v


def rgb2gray(img):
    img_gray = np.double(
        (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 0] + 0.1140 * img[:, :, 0]))  # RGB to Gray Conversion
    return img_gray


def run():
    window_width = 780
    window_heigth = 680

    # frame0 is used to place contents in main code
    frame0 = Frame(root, width=window_width, height=window_heigth)
    frame0['borderwidth'] = 2
    frame0['relief'] = 'sunken'
    frame0.place(x=10, y=20)

    canvas = Canvas(frame0, width=750, height=600, bg="white")
    image0 = ImageTk.PhotoImage(Image.open("image.jpg"))
    canvas.create_image(0, 50, anchor=NW, image=image0)
    canvas.pack()
    # add this below commented codes to add further frames to display

    # Frame2 is used to add the contents in home Page
    frame2 = Frame(root, width=window_width, height=window_heigth)
    frame2['borderwidth'] = 2
    frame2['relief'] = 'sunken'
    frame2.place(x=10, y=20)

    canvas = Canvas(frame2, width=750, height=600, bg="midnight blue")
    image = ImageTk.PhotoImage(Image.open("image.jpg"))
    canvas.create_image(0, 50, anchor=NW, image=image)
    canvas.pack()
    frame2.place(x=10, y=20)
    frame0.place_forget()

    # while clicking on runway image detection in home page, following browse function will invoke
    def browse():
        frame0.place(x=10, y=20)
        # frame1.place_forget()
        frame2.place_forget()

        # while clicking on browse button in 2nd page, following browze function will invoke
        def browze():

            browz = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select IMage File",
                                               filetypes=(("JPG file", "*.jpg"),
                                                          ("PNG file", "*.png"), ("All file", "*.*")))

            lo = Image.open(browz)
            load = lo.resize((350, 200))

            render = ImageTk.PhotoImage(load)
            la = Label(frame0, image=render, bg="slate blue")

            la.image = render
            la.place(x=250, y=60)

            def submit_dehaze():
                # if (condition that satisfies a runway image):
                # messagebox.showwarning("Result", "Sorry !!, Runway is not found")
                # else

                # include above messagebox in else block

                f_c = cv2.imread(browz)

                S = 19  # Defines the size of sliding-window (SxS) for Airlight estimation

                resize_img = cv2.resize(f_c, (640, 480))

                img = resize_img

                f_c = np.double(img)  # Read Input Hazy Image

                ####################
                # Size of the Input Image
                Nr, Nc, Np = f_c.shape

                #########################################
                # Extention of the input image to avoid edge issues
                A1 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1)
                A2 = np.concatenate((np.fliplr(f_c), f_c, np.fliplr(f_c)), axis=1)
                A3 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1)
                f_proc = np.concatenate((A1, A2, A3), axis=0)
                f_proc = f_proc[Nr - int((S - 1) / 2):2 * Nr + int((S - 1) / 2),
                         Nc - int((S - 1) / 2):2 * Nc + int((S - 1) / 2), :]

                A_test = np.zeros([Nr, Nc])
                f_mv = np.zeros([S, S])
                K = np.floor(2 * S * S / 3)
                for k in range(Nr):
                    leyend = 'Estimating Airlight: ' + str(int(100 * (k + 1) / Nr)) + '%'
                    print(leyend)
                    for l in range(Nc):
                        f_mv = (f_proc[k:S + k, l:S + l])
                        f_max = f_mv.max()
                        f_min = f_mv.min()
                        # u = np.mean(f_mv[:])
                        u = (f_min + f_max) / 2.0
                        # v = (1 + np.var(f_mv[:]))
                        v = f_max - f_min
                        A_test[k, l] = u / (1 + v)
                print('Done!')

                x0, y0 = np.where(A_test == A_test.max())
                A = np.zeros(3)
                A[0] = f_c[x0[0], y0[0], 0]
                A[1] = f_c[x0[0], y0[0], 1]
                A[2] = f_c[x0[0], y0[0], 2]
                A_est = 0.2989 * A[0] + 0.5870 * A[1] + 0.1140 * A[2]

                t_est = np.zeros([Nr, Nc])
                trans = np.zeros([Nr, Nc])

                # PAR = [15, 0.01, 1.0, 4.] # Parameters for maguey
                PAR = [9, 0.01, 10.0, 4.0]  # Parameters for flores
                # PAR = [19, 0.6, 0.6, 6.] # Parameters for fuente

                S = PAR[0]  # Defines the size of sliding-window SxS
                w = PAR[1]  # Minimum allowed value of transmission
                j0 = PAR[2]  # Parameter for transmission estimation
                Kdiv = PAR[3]  # Parameter for caculation of K in adaptive neighborhoods

                y = np.zeros([Nr, Nc, Np])
                K = S ** 2 - int((S ** 2) / Kdiv)

                for k in range(Nr):
                    leyend = 'Estimating Transmission: ' + str(int(100 * (k + 1) / Nr)) + '%'
                    print(leyend)
                    for l in range(Nc):
                        f_w = f_proc[k:S + k, l:S + l, :]
                        f_v = KNB(f_w, K)
                        Fmax = f_v.max()
                        Fmin = f_v.min()
                        range_fv = Fmax - Fmin
                        fv_avg = (Fmin + Fmax) / 2.0
                        if range_fv < w:
                            range_fv = w
                        alpha = range_fv / (j0 * A_est)
                        t_est[k, l] = (A_est - (alpha * fv_avg + (1 - alpha) * Fmin)) / (A_est - alpha * fv_avg)

                        if t_est[k, l] > 1:
                            t_est[k, l] = 1
                        if t_est[k, l] < w:
                            t_est[k, l] = w
                print('Done!')

                # trans = denoise_tv_chambolle(t_est, weight=1000.0, multichannel=False)
                trans = guidedFilter(np.uint8(f_c), np.uint8(255 * t_est), 30, 0.1) / 255.0

                y[:, :, 0] = (f_c[:, :, 0] - A[0]) / trans + A[0]
                y[:, :, 1] = (f_c[:, :, 1] - A[1]) / trans + A[1]
                y[:, :, 2] = (f_c[:, :, 2] - A[2]) / trans + A[2]
                y[:, :, :] = abs(y[:, :, :])
                xs, ys, zs = np.where(y > 255)
                y[xs, ys, zs] = 255.0

                plt.subplot(121), plt.imshow(f_c / 255.), plt.title('Hazy Image')
                plt.subplot(122), plt.imshow(y / 255.), plt.title('Processed Image')
                # plt.savefig("plot_new.png")
                # plt.imshow
                # plt.show()
                plt.imsave('dehaze.jpg', np.uint8(y))

                crr = Image.open('dehaze.jpg')
                cr = crr.resize((300, 270))
                crp = ImageTk.PhotoImage(cr)
                img1 = Label(frame0, image=crp)
                img1.image = crp
                img1.place(x=50, y=300)

            # wait for some time to display the result

            def submit_localise():
                image = cv2.imread("dehaze.jpg")
                img = cv2.resize(image, (24, 24))

                model = load_model("runway_weights.hdf5")

                def cnnPreprocess(img):
                    img = img.astype('float32')
                    img /= 255
                    img = np.expand_dims(img, axis=2)
                    img = np.expand_dims(img, axis=0)

                    return img

                grayco = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                prediction = (model.predict(cnnPreprocess(grayco)))

                print(prediction)

                if prediction[[0]] >= 0.99:
                    messagebox.showwarning("Result", "Runway Image Detected")
                    img = cv2.imread("dehaze.jpg")
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(image.shape)
                    height = image.shape[0]
                    width = image.shape[1]
                    region_of_interest_vertices = [
                        (0, height),
                        (width / 2, height / 2),
                        (width, height)
                    ]
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    canny_image = cv2.Canny(gray_image, 100, 200)
                    cropped_image = region_of_interest(canny_image,
                                                       np.array([region_of_interest_vertices], np.int32), )
                    lines = cv2.HoughLinesP(cropped_image,
                                            rho=3,
                                            theta=np.pi / 180,
                                            threshold=100,
                                            lines=np.array([]),
                                            minLineLength=30,
                                            maxLineGap=10)
                    image_with_lines = drow_the_lines(image, lines)
                    # plt.imshow(image_with_lines)
                    # plt.show()

                    plt.imsave("localise.png", image_with_lines)

                    crr = Image.open('localise.png')
                    cr = crr.resize((300, 270))
                    crp = ImageTk.PhotoImage(cr)
                    img1 = Label(frame0, image=crp)
                    img1.image = crp
                    img1.place(x=430, y=300)

                else:

                    messagebox.showwarning("Result", "No Runway Detected")

            def localise():

                image = cv2.imread(browz)
                img = cv2.resize(image, (24, 24))

                model = load_model("runway_weights.hdf5")

                def cnnPreprocess(img):
                    img = img.astype('float32')
                    img /= 255
                    img = np.expand_dims(img, axis=2)
                    img = np.expand_dims(img, axis=0)

                    return img

                grayco = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                prediction = (model.predict(cnnPreprocess(grayco)))

                print(prediction)

                if prediction[[0]] >= 0.99:
                    messagebox.showwarning("Result", "Runway Image Detected")
                    img = cv2.imread(browz)
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(image.shape)
                    height = image.shape[0]
                    width = image.shape[1]
                    region_of_interest_vertices = [
                        (0, height),
                        (width / 2, height / 2),
                        (width, height)
                    ]
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    canny_image = cv2.Canny(gray_image, 100, 200)
                    cropped_image = region_of_interest(canny_image,
                                                       np.array([region_of_interest_vertices], np.int32), )
                    lines = cv2.HoughLinesP(cropped_image,
                                            rho=3,
                                            theta=np.pi / 180,
                                            threshold=100,
                                            lines=np.array([]),
                                            minLineLength=30,
                                            maxLineGap=10)
                    image_with_lines = drow_the_lines(image, lines)
                    # plt.imshow(image_with_lines)
                    # plt.show()

                    plt.imsave("localise.png", image_with_lines)

                    crr = Image.open('localise.png')
                    cr = crr.resize((280, 180))
                    crp = ImageTk.PhotoImage(cr)
                    img1 = Label(frame0, image=crp)
                    img1.image = crp
                    img1.place(x=500, y=300)


                else:

                    messagebox.showwarning("Result", "No Runway Detected")





            b3 = Button(frame0, text="Localisation", bg="slate blue", command=submit_localise, width=20, height=10)
            b3.place(x=290, y=300)
            b2 = Button(frame0, text="Dehaze_image", bg="slate blue", command=submit_dehaze, width=20, height=10)
            b2.place(x=50, y=300)
            b4 = Button(frame0, text="Localise_input_image", bg="slate blue", command=localise, width=20, height=10)
            b4.place(x=500, y=300)

        b6 = Button(frame0, text="<<Back", bg="slate blue", fg="white", width=10, bd=15, command=run,
                    font=("arial", 11, "bold"))
        b6.place(x=45, y=200)
        b5 = Button(frame0, text='Browse', font=('arial', 11, 'bold'), height=1, bd=15, width=10, fg="white",
                    bg="slate blue", command=browze).place(x=45, y=100)

    lbl1 = Label(root, text="RUNWAY DETECTION SYSTEM", fg="white", bg="slate blue", width=50, cursor="circle",
                 font=('Arial', 18, 'italic'))
    lbl1.place(x=30, y=30)
    b1 = Button(frame2, text='Runway Image Detection', font='arial 15', command=browse, bg="slateblue", fg="white",
                bd=15, width=50, height=1).place(x=100, y=500)
    root.mainloop()


run()