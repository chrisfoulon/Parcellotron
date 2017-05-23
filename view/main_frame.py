# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap
# I know I know ...
from PyQt5.QtWidgets import *

class MainFrame(QMainWindow):
    """ Main parcellation frame
    This is the main container of all the components of the GUI.
    Parameters
    ----------
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.title = 'Parcellotron'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
        self.center()

    def initUI(self):
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Menu')
        fileMenu.addAction(exitAction)

        listWidget = QComboBox()

        #Resize width and height
        listWidget.resize(50,120)

        listWidget.addItem("Tractography 4D");
        listWidget.addItem("Tractography matrix");
        # listWidget.addItem("Item 3");
        # listWidget.addItem("Item 4");

        self.root = QRadioButton("Root folder")
        self.root.setChecked(True)
        self.group_check = QCheckBox("Group level analysis")
        # self.root.toggled.connect(lambda:self.btnstate(self.root))

        self.subj = QRadioButton("Subject folder")
        # self.subj.toggled.connect(lambda:self.btnstate(self.subj))
        self.root_bro = BrowseWidget(None, "dir", "")
        self.subj_bro = BrowseWidget(None, "dir", "")

        seed_lbl = QLabel("Select the prefix of you seed file(s):")
        self.seed_fld = QLineEdit(self)
        target_lbl = QLabel("Select the prefix of you target file(s):")
        self.target_fld = QLineEdit(self)

        grid = QGridLayout()
        grid.addWidget(self.root, 0, 0)
        grid.addWidget(self.group_check, 0, 1)
        grid.addWidget(self.root_bro, 1, 0)
        grid.addWidget(self.subj, 2, 0)
        grid.addWidget(self.subj_bro, 3, 0)
        grid.addWidget(seed_lbl, 4, 0)
        grid.addWidget(self.seed_fld, 5, 0)
        grid.addWidget(target_lbl, 4, 1)
        grid.addWidget(self.target_fld, 5, 1)
        grid.addWidget(QLabel("Select the modality:"), 6, 0)
        grid.addWidget(listWidget, 7, 0)

        # root_lay = QHBoxLayout()
        # root_lay.addWidget(self.root)
        # root_lay.addWidget(self.group_check)
        #
        # plus_bro_lay = QVBoxLayout()
        # plus_bro_lay.addLayout(root_lay)
        # plus_bro_lay.addWidget(self.root_bro)
        #
        # radio_lay = QVBoxLayout()
        # radio_lay.addLayout(root_lay)
        # radio_lay.addWidget(self.subj)

        vBoxlayout = QVBoxLayout()
        vBoxlayout.addLayout(grid)
        vBoxlayout.addStretch(1)

        self.tab1.setLayout(vBoxlayout)

        self.tabs.addTab(self.tab1,"Files")
        self.tabs.addTab(self.tab2,"Parcellation")
        self.setCentralWidget(self.tabs)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.statusBar().showMessage('Not all parameters have been set')
        self.show()


    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class BrowseWidget(QWidget):
    """
    class BrowseWidget
    Widget composé d'un LineEdit et d'un bouton qui au clic ouvrira
    un QFileDialog
    :param mode : permet de choisir le filtre de type de fichier :
        -"dir" pour un répertoire
        -"xls" pour un fichier .xls
    """

    def __init__(self, parent=None, mode="dir", label="", wid=200,
                 hei=40, defDir="",):
        super(BrowseWidget, self).__init__(parent)
        # mode
        self.mode = mode

        # Répertoire par défaut
        self.defDir = ""

        self.lbl = QLabel(label)
        self.lbl.setAlignment(Qt.AlignHCenter)

        # Passage de la font en bold !
        font = QFont()
        font.setBold(True)
        self.lbl.setFont(font)

        # Les composants du widget
        # Un layout
        self.lay = QGridLayout()
        # Les composants
        self.fld = QLineEdit(self)
        self.but = QPushButton()
        Icon = QPixmap("lil_folder.png")
        # On agrandi l'icone pour ensuite resize pour enlever les bordures
        # récalcitrantes
        self.but.setIconSize(QSize(30,30))
        self.but.setFixedSize(24,24)
        self.but.setIcon(QIcon(Icon))

        # Placement des composant dans la layout du widget
        row = 0
        if self.lbl.text() != "":
            self.lay.addWidget(self.lbl, 0, 0, 1, 2)
            row = 1
        self.lay.addWidget(self.fld, row, 0)
        self.lay.addWidget(self.but, row, 1)

        self.setLayout(self.lay)
        #On fixe la taille du Widget entier
        self.setFixedSize(wid, hei)

        # Partie interaction

        # Ouverture du browser
        self.but.clicked.connect(self.browse)

        self.setDefDir(defDir)

    def browse(self):
        """
        browse
        Crée un QFileDialog permettant de sélectionner un fichier ou
        un répertoire selon le mode.
        """
        path = ""
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Open Directory",
                                                    self.defDir)
        elif self.mode == "xls":
            # La fonction renvoie un tuple donc il faut prendre le premier elt.
            if screen.replaceRes():
                path = QFileDialog.getSaveFileName(
                    self, "Select File", self.defDir, "*.xls",
                    options=QFileDialog.DontConfirmOverwrite)[0]
            else:
                path = QFileDialog.getSaveFileName(self, "Select File",
                                                   self.defDir, "*.xls")[0]
            # Si dans le browser on entre un fichier qui n'a pas l'extension
            # .xls, lève une erreur
            if path != "" and path.split(".")[-1] != "xls":
                dia = criticalDial(self, "Error !", "Invalid results file")
                return
        else:
            print("Error : the mode doesn't exists")

        if path != "":
            # On vérifier qu'il n'y a pas d'espaces dans le path
            if not path.__contains__(" "):
                if self.mode == "xls":
                    f = QFile(path)
                    if not f.open(QIODevice.ReadWrite):
                        dia = criticalDial(self, "Error !",
                                           "Invalid path or permission denied")
                        return
                    else:
                        f.close()
                self.fld.setText(path)
            else:
                dia = criticalDial(
                    self, "Error !", "The path : " + path + " contains spaces")

    # Partie interaction
    def getText(self):
        return self.fld.text()

    def setText(self, text=""):
        self.fld.setText(text)

    def setDefDir(self, defDir=""):
        self.defDir = defDir

    # Fonctions de cohérences

    # Cohérence champ de texte et la variable path
    def pathEqFld(self):
        return self.getText() == self.path

def criticalDial(parent=None, title="Error !", message="",
                 wid=300, hei=300):
    dia = QMessageBox(parent)
    # ne fonctionne pas
    # dia.setMinimumSize(wid, hei)
    dia.addButton("OK", QMessageBox.AcceptRole)
    dia.critical(parent, title, message)
    return dia

class SettingsFrame(QDialog):
    """
    class SettingsFrame
    Widget permettant de modifier les paramètres de l'application
    """
    def __init__(self, conf, parent=None):
        super(SettingsFrame, self).__init__(parent)
        self.setWindowTitle("Settings")
        # La fenêtre
        self.setFixedSize(FRAME_W, FRAME_H)

        # TractoConfig récupéré en paramètre
        self.conf = conf

        self.mainLay = QVBoxLayout()
        self.lay = QGridLayout()

        # Un layout pour la séléction du chemin par défaut
        self.defLay = QGridLayout()
        self.checkDef = QCheckBox()
        self.defBro = BrowseWidget(None, "dir", "Set Path :", 200)
        self.defLay.addWidget(self.checkDef, 1, 0, Qt.AlignTop)
        self.defLay.addWidget(self.defBro, 0, 1, 2, 1)

        # Une checkbox pour savoir si on sauvegarde les modifications des paths
        self.savePath = QCheckBox("Save directories after shutdown")
        font = QFont()
        font.setBold(True)
        self.savePath.setFont(font)

        # Les boutons de reset aux paths par défaut
        self.resetLes = QPushButton("Restore default lesion directory")
        self.resetTra = QPushButton("Restore default tract directory")
        self.resetRes = QPushButton("Restore default result file")

        # Bouton de fermeture
        self.closeButton = QPushButton("Close")
        self.closeButton.setMinimumHeight(20)



        # Ajout des composant au layout principal
        self.lay.setAlignment(Qt.AlignTop)
        self.lay.addLayout(self.defLay, 0, 0)
        self.lay.addWidget(self.savePath, 1, 0)
        self.lay.addWidget(self.resetLes, 2, 0)
        self.lay.addWidget(self.resetTra, 3, 0)
        self.lay.addWidget(self.resetRes, 4, 0)

        self.mainLay.addLayout(self.lay)
        self.mainLay.addWidget(self.closeButton)

        self.setLayout(self.mainLay)

        self.checkDef.setFixedWidth(15)

        # Déplacement au centre
        self.move(QApplication.desktop().screen().rect().center()
                  - self.rect().center())

        self.initParam()

        self.resetLes.clicked.connect(parent.resetLes)
        self.resetTra.clicked.connect(parent.resetTra)
        self.resetRes.clicked.connect(parent.resetRes)

        self.closeButton.clicked.connect(self.close)

    def closeEvent(self, event):
        if self.isDefDirChecked() and self.defBro.getText() == "":
            criticalDial(message="The path is not specified")
        elif not QDir(self.defBro.getText()).exists():
            criticalDial(message="Invalid path")
        else:
            if self.isDefDirChecked():
                self.conf.saveDefDir(self.defBro.getText())
                screen.setDefDir()
            else:
                self.conf.settings.remove("defDir")
                screen.setDefDir()
            # C'est redondant mais c'est plus clair
        if self.isSaveCheck():
            self.conf.savePaths(save=True)
        else:
            self.conf.savePaths(save=False)
        event.accept()

    def isDefDirChecked(self):
        return self.checkDef.isChecked()

    def isSaveCheck(self):
        return self.savePath.isChecked()

    def initParam(self):
        tmp = self.conf.getValue("defDir")
        if tmp != "":
            self.checkDef.setChecked(True)
            self.defBro.setText(tmp)
        if self.conf.getValue("savePaths", bool) is True:
            self.savePath.setChecked(True)
