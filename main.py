# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QStyleFactory, QApplication)

if __name__ == '__main__':
    import sys
    import view.main_frame as mf

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("fusion"))

    main_frame = mf.MainFrame()
    main_frame.show()

    sys.exit(app.exec_())
