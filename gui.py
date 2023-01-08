import sys
import pickle

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from copy import deepcopy

from preprocessing import *
from network import *

instances = {
    "alpha" : 0,
    "beta" : 0,
    "gamma" : 0,
    "delta" : 0,
    "epsilon" : 0
}

curr_instance = list()
raw_dataset = list()

labels_prediction = list(instances.keys())
labels = iter(instances.keys())
curr_label = next(labels)

prediction = {
    'label' : ""
}
           
class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.pressed = False
        self.dots = list()
        self.predict_flag = False
        
    def initUI(self):
        
        self.label = QLabel(self)
        self.canvas = QPixmap(800, 800)
        self.canvas.fill(Qt.white)
        self.label.setPixmap(self.canvas)        
        
        # Buttons
        next_symbol_button = QPushButton('Next Symbol')
        next_symbol_button.clicked.connect(self.next_symbol_clicked)
        train_button = QPushButton('Train')
        train_button.clicked.connect(self.train_clicked)
        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.predict_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(next_symbol_button)
        hbox.addWidget(train_button)
        hbox.addWidget(predict_button)    # TODO: What to do with this one?
        # hbox.addWidget(labels_counter, stretch=0, alignment=Qt.AlignTop | Qt.AlignCenter)
        # hbox.addWidget(labels_counter, stretch=0)

        # Label counters and prediction
        self.labels_counter = QLabel(self)        
        self.labels_counter.setText(f"Alpha: {instances['alpha']}   Beta: {instances['beta']}   Gamma {instances['gamma']}   Delta: {instances['delta']}   Epsilon: {instances['epsilon']}" )
        
        self.prediction = QLabel(self)
        self.prediction.setText(f"Prediction: {prediction['label']}")

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)  # Adding the white canvas to draw on.
        vbox.addWidget(self.labels_counter, stretch=0, alignment=Qt.AlignBottom)    # Adding the labels counter to the bottom
        vbox.addWidget(self.prediction)

        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.setGeometry(400, 100, 800, 800)
        self.setWindowTitle('Box layout example, QHBoxLayout, QVBoxLayout')  
        self.show()
    
    def mouseMoveEvent(self, event):
        # self.label.setText('Mouse coords: ( %d : %d )' % (event.x(), event.y()))

        if self.pressed:
            # IF LEN < 20: REJECT; ELSE: ACCEPT; THEN SAVE AND CLEAR self.dots
            self.dots.append((event.x(), event.y()))
            
            if len(self.dots) > 1:
                x1, y1 = self.dots[-2]
                x2, y2 = self.dots[-1]
                
                painter = QPainter(self.label.pixmap())
                pen = QPen(Qt.red, 2)
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)
                painter.end()
                self.update()   # Calls the paintEvent method
            # print(f"'Mouse coords: ( {event.x()}, {event.y()} )")
            
            
    def next_symbol_clicked(self):
        global curr_label
        
        if curr_label == "epsilon":
            print("This is the last symbol! Start training!")
        else:
            curr_label = next(labels)

    def train_clicked(self):
        # global curr_label
        # if curr_label != "epsilon":
        #     proceed = input("Warning! Not all labels have instances. Continue? [y|n]")
        #     if proceed == 'n':
        #         print(f"You may continue sampling. You are currently at {curr_label}.")
        #         return
            
        # Saving data to save time.
        # with open("raw_dataset.pickle", "wb") as file:
        #     pickle.dump(raw_dataset, file)
        
        # with open("raw_dataset.pickle", "rb") as file:
        #     ds = pickle.load(file)
        
        print("Training started.")
        # Training magic ... **.
        # dataset = preprocess_data(raw_dataset)  # Dataset ready for training NOTE: I have a ds ready and preprocessed.

        print("Training finished. The model is ready to predict.")
        pass
        
    def predict_clicked(self):
        
        with open("extended_prepared_dataset.pickle", "rb") as file:
            ds = pickle.load(file)

        X = [el[0] for el in ds]
        Y = [el[1] for el in ds]
        M = X[0].shape[0]
        # print(X[0], Y[0])
        # print(M)

        arh = "20s20s"
        self.net = Network(M, arh)
        
        self.net.train(X, Y, epochs=100, alg=1)
        self.predict_flag = True
        # net.train_minibatch(X, Y)
        

    # def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
    def mousePressEvent(self, a0: QMouseEvent) -> None:
        self.pressed = True
        self.label.setPixmap(self.canvas)
        
    # def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        global curr_label
        
        if len(self.dots) > 80:
            raw_dataset.append( ( deepcopy(self.dots), curr_label) )
            # Updating the number of instances per label
            instances[curr_label] += 1
            self.labels_counter.setText(f"Alpha: {instances['alpha']}   Beta: {instances['beta']}   Gamma {instances['gamma']}   Delta: {instances['delta']}   Epsilon: {instances['epsilon']}" )            

        if self.predict_flag:
            x = preprocess_symbol_vector(deepcopy(self.dots))
            prediction['label'] = self.net.predict(x)
            self.prediction.setText(f"Prediction: {prediction['label']}")
        
        self.dots.clear()

        self.pressed = False

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
