import sys
sys.path.append('./GUI')
from gui import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QMessageBox, QSizePolicy, QFileDialog
import sys, os
import train
import deal
from skimage import io, transform


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.statusbar.showMessage('Designed by WangYufei')

    def openfile(self):
        self.fileName, self.filetype = QFileDialog.getOpenFileName(self,
                                                                   "选取待识别图片",
                                                                   "./",
                                                                   "*jpg (*.jpg)")  # 设置文件扩展名过滤,注意用双分号间隔'''

        if self.fileName:
            img_raw = io.imread(self.fileName)
            img = deal.detect_deal(img_raw)
            img_raw = transform.resize(img_raw, (200, 200))
            io.imsave('temp_raw.jpg', img_raw)
            self.img_show.setPixmap(QtGui.QPixmap('temp_raw.jpg'))
            io.imsave('temp.jpg', img)
            self.img_show2.setPixmap(QtGui.QPixmap('temp.jpg'))
            str1 = "Dealing the image " + self.fileName + '\n'
            result = train.forecast('temp.jpg')
            str2 = "The result is " + str(result[0]) + '\n'
            result = list(result[0])
            idx = result.index(max(result))
            if idx == 0:
                str3 = '眼部健康' + '可能性为' + str(max(result))
            else:
                str3 = '存在眼部疾病。疾病最可能为' + str(idx) + '期，可能性为' + str(max(result))
            self.textBrowser.setText(str1 + str2 + str3)
            print(type(result))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
