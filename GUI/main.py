from gui import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication,QWidget,QMainWindow,QMessageBox,QSizePolicy,QFileDialog
import sys,os
from train import *

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.statusbar.showMessage('Designed by WangYufei')

    def openfile(self):
        self.fileName, self.filetype = QFileDialog.getOpenFileName(self,
                                                          "选取待识别图片",
                                                          "./",
                                                          "*All file (*.*)")  # 设置文件扩展名过滤,注意用双分号间隔'''
        print(self.fileName,self.filetype)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
