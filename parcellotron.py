# -*- coding: utf-8 -*-
import sys

from PyQt5.QtWidgets import (QStyleFactory, QApplication)

import view.main_frame as mf

if __name__ == '__main__':

    app = QApplication(sys.argv)
    # app.setStyle(QStyleFactory.create('GTK+'))

    main_frame = mf.MainFrame()
    main_frame.show()

    sys.exit(app.exec_())
