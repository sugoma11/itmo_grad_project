import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from random import choice
from numpy import any
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.preprocessing import StandardScaler


class ImageAnnotations3D():
    def __init__(self, xyz, imgs, ax3d, ax2d, flag=None):
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        if flag == 'images':
            for s, im in zip(self.xyz, self.imgs):
                x, y = self.proj(s)
                self.annot.append(self.image(im, [x, y], ax2d))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event", self.update)

        self.funcmap = {"button_press_event": self.ax3d._button_press,
                        "motion_notify_event": self.ax3d._on_move,
                        "button_release_event": self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                    for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1,
            calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self, arr, xy, ax):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=1.5)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy,
                                      xycoords='data', boxcoords="offset points", pad=0.01)
        self.ax2d.add_artist(ab)
        return ab

    def update(self, event):
        if any(self.ax3d.get_w_lims() != self.lim) or \
                any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s, ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)


class VisExtract():

    def filer(self):
        scaler = StandardScaler()
        cnt = self.num
        filelist = os.listdir(f'{self.file_name}')
        while len(filelist) != 0:
            file = choice(filelist)
            img = cv2.imread(f'{self.file_name}/{file}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_CUBIC)

            if file[0] == '1':
                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (255, 0, 0), thickness=50,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.xs.append(img[:, :, 0].mean())
                self.ys.append(img[:, :, 1].mean())
                self.zs.append(img[:, :, 2].mean())
                self.y.append(int(file[0]))

            else:
                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (0, 255, 0), thickness=50,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.xs.append(img[:, :, 0].mean())
                self.ys.append(img[:, :, 1].mean())
                self.zs.append(img[:, :, 2].mean())
                self.y.append(int(file[0]))

            cnt -= 1
            # os.remove(f'{file_name}/{file}')
            filelist.remove(file)
            if cnt == 0:
                self.xs, self.ys, self.zs = np.array(self.xs),  np.array(self.ys), np.array(self.zs)

                self.xs, self.ys, self.zs = scaler.fit_transform(self.xs.reshape(-1, 1)),\
                                            scaler.fit_transform(self.ys.reshape(-1, 1)),\
                                            scaler.fit_transform(self.zs.reshape(-1, 1))

                self.y = np.array(self.y).reshape(-1, 1)
                self.data = np.c_[self.xs, self.ys, self.zs]
                break

    def __init__(self, file_name, num, action):

        self.imgs = []
        self.action = action
        self.data = None

        # R, G, B
        self.xs = []
        self.ys = []
        self.zs = []

        self.y = []

        self.file_name = file_name
        self.num = num

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=Axes3D.name)

        self.filer()
        self.plotter()

    @staticmethod
    def f_xy(a, b, c, d, x, y):
        return - x * a / c - y * b / c - d / c

    def add_hyperplane(self, params):

        x = np.arange(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 0.5)
        y = np.arange(np.min(self.data[:, 1]), np.max(self.data[:, 1]), 0.5)

        x, y = np.meshgrid(x, y)
        eq = VisExtract.f_xy(params[0], params[1], params[2], params[3], x, y)

        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.plot_surface(x, y, eq, cmap='spring', alpha=0.95)

        ax_3d.set_xlabel('R channel')
        ax_3d.set_ylabel('G channel')
        ax_3d.set_zlabel('B channel')

        ax_3d.scatter(self.xs[self.pos], self.ys[self.pos], self.zs[self.pos], alpha=1, color='r', marker='x')
        ax_3d.scatter(self.xs[self.neg], self.ys[self.neg], self.zs[self.neg], alpha=1, color='g', marker='o')
        plt.show()

    def plotter(self, flag=None):

        self.pos = np.where(self.y == 1)
        self.neg = np.where(self.y == 0)

        self.ax.scatter(self.xs[self.pos], self.ys[self.pos], self.zs[self.pos], alpha=0, color='r', marker='x')
        self.ax.scatter(self.xs[self.neg], self.ys[self.neg], self.zs[self.neg], alpha=0, color='g', marker='o')

        self.ax2 = self.fig.add_subplot(111, frame_on=False)
        self.ax2.axis("off")
        self.ax2.axis([0, 1, 0, 1])

        ia = ImageAnnotations3D(self.data, self.imgs, self.ax, self.ax2, 'images')

        self.ax.set_xlabel('R channel')
        self.ax.set_ylabel('G channel')
        self.ax.set_zlabel('B channel')

        # ax.view_init(30, 180)

        if self.action == 'show':
            plt.show()

        elif self.action == 'save':
            plt.savefig(f'{np.random.randint(0, 10)}')


# tst = VisExtract('data', 100, 'show')
#
# (trainData, testData, trainLabels, testLabels) = train_test_split(tst.data, tst.y, test_size=0.25, random_state=9)
# model = LogisticRegression(random_state=0, solver='lbfgs').fit(trainData, trainLabels)
#
# tst.add_hyperplane(list((model.coef_[0][0], model.coef_[0][1], model.coef_[0][2], model.intercept_[0])))
