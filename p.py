from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ===================== Dataset Preparation =====================
def prepare_dataset():
    data = []
    labels = []
    classes = 5  # black, cinder, laterite, peat, yellow
    cur_path = os.getcwd()
    class_names = ['black_soil', 'cinder_soil', 'laterite_soil', 'peat_soil', 'yellow_soil']

    print("Obtaining Images & Labels...")
    for i in range(classes):
        path = os.path.join(cur_path, 'dataset/train', str(i))
        if not os.path.exists(path):
            print(f"Path {path} does not exist!")
            continue
        images = os.listdir(path)
        for a in images:
            try:
                image_path = os.path.join(path, a)
                test_image = Image.open(image_path).resize((64, 64))
                # Convert to RGB if needed
                if test_image.mode != 'RGB':
                    test_image = test_image.convert('RGB')
                
                data.append(np.array(test_image))
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {a}: {str(e)}")

    print(f"Dataset Loaded: {len(data)} images")
    data = np.array(data).astype('float32') / 255.0
    labels = np.array(labels)
    
    return data, labels, class_names

# ===================== PyQt GUI =====================
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Plant Detection with Soil Information")
        MainWindow.resize(900, 650)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # ---- Header ----
        self.label_2 = QtWidgets.QLabel("🌱 Plant Detection with Soil Information")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont("Courier New", 16, QtGui.QFont.Bold)
        self.label_2.setFont(font)
        layout.addWidget(self.label_2)

        # ---- Image Display ----
        self.imageLbl = QtWidgets.QLabel()
        self.imageLbl.setFixedSize(400, 300)
        self.imageLbl.setStyleSheet("border: 2px solid gray; background: #fafafa;")
        self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.imageLbl, alignment=QtCore.Qt.AlignCenter)

        # ---- Buttons ----
        btnLayout = QtWidgets.QHBoxLayout()
        self.BrowseImage = QtWidgets.QPushButton("📂 Browse Image")
        self.Classify = QtWidgets.QPushButton("🔍 Classify")
        self.Training = QtWidgets.QPushButton("⚙️ Train Model")
        for btn in [self.BrowseImage, self.Classify, self.Training]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
            """)
            btnLayout.addWidget(btn)
        layout.addLayout(btnLayout)

        # ---- Result Section ----
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setFixedHeight(50)
        self.textEdit.setPlaceholderText("Predicted Soil Type will appear here...")
        layout.addWidget(self.textEdit)

        # ---- Soil Information ----
        self.PlantDetails = QtWidgets.QTextEdit()
        self.PlantDetails.setPlaceholderText("Soil details will appear here after classification...")
        self.PlantDetails.setReadOnly(True)
        layout.addWidget(self.PlantDetails)

        MainWindow.setCentralWidget(self.centralwidget)

        # Connect Buttons
        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)
        
        # Initialize variables
        self.file = None
        self.class_names = ['black_soil', 'cinder_soil', 'laterite_soil', 'peat_soil', 'yellow_soil']

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)

    def classifyFunction(self):
        if not os.path.exists('my_model.h5'):
            self.textEdit.setText("⚠️ Model not found! Please train the model first.")
            return
            
        if not self.file:
            self.textEdit.setText("⚠️ Please select an image first!")
            return

        try:
            model = load_model("my_model.h5")
            test_image = Image.open(self.file).resize((64, 64))
            if test_image.mode != 'RGB':
                test_image = test_image.convert('RGB')
                
            test_image = np.array(test_image).astype('float32') / 255.0
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)[0]
            predicted_class = np.argmax(result)
            confidence = result[predicted_class]
            Plant_Name = self.class_names[predicted_class]
            
            self.textEdit.setText(f"✅ Predicted Soil: {Plant_Name} (Confidence: {confidence:.2f})")

            # Detailed soil info with extended crops
            soil_info = {
                "black_soil": "🌑 Black Soil:\nCrops: Cotton, Wheat, Pulses, Soybean, Maize, Sugarcane, Groundnut\nSoil: Rich in minerals, retains moisture\nBioFertilizer: FYM 5-10kg + Azospirillum",
                "cinder_soil": "🪨 Cinder Soil:\nCrops: Tomato, Brinjal, Chilli, Capsicum, Okra, Beans, Cabbage, Carrot, Radish, Beetroot, Potato, Onion, Garlic, Ginger, Turmeric\nSoil: Well-drained\nBioFertilizer: Compost + PSB + Vermicompost",
                "laterite_soil": "🟤 Laterite Soil:\nCrops: Cashew, Coffee, Coconut, Areca nut\nSoil: Rich in iron & aluminum\nBioFertilizer: Trichoderma, Azotobacter",
                "peat_soil": "🟢 Peat Soil:\nCrops: Carrot, Radish, Beetroot, Mushrooms, Lettuce, Spinach\nSoil: High organic matter, moisture-retentive\nBioFertilizer: Rhizobium, PSB, Vermicompost",
                "yellow_soil": "🟡 Yellow Soil:\nCrops: Wheat, Maize, Pulses, Sorghum, Groundnut\nSoil: Well-drained\nBioFertilizer: Azospirillum, PSB"
            }
            self.PlantDetails.setText(soil_info.get(Plant_Name, "Soil details not available yet."))
            
        except Exception as e:
            self.textEdit.setText(f"❌ Error during classification: {str(e)}")

    def trainingFunction(self):
        self.textEdit.setText("⚙️ Preparing dataset...")
        QtWidgets.QApplication.processEvents()  # Update UI
        
        try:
            # Prepare dataset
            data, labels, self.class_names = prepare_dataset()
            
            if len(data) == 0:
                self.textEdit.setText("❌ No images found in dataset! Please check your dataset path.")
                return
                
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            
            # Convert labels to categorical
            y_train = to_categorical(y_train, len(self.class_names))
            y_test = to_categorical(y_test, len(self.class_names))
            
            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(X_train)
            
            self.textEdit.setText("⚙️ Training in progress...")
            QtWidgets.QApplication.processEvents()  # Update UI

            # Build model with improved architecture
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
            model.add(MaxPool2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPool2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPool2D((2, 2)))
            model.add(Dropout(0.3))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(self.class_names), activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Train with validation
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                steps_per_epoch=len(X_train) // 32,
                validation_data=(X_test, y_test),
                epochs=1000,  # Reduced epochs for faster training
                verbose=2  # Set to 1 if you want to see progress in console
            )
            
            # Save model
            model.save("my_model.h5")
            
            # Evaluate model
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            self.textEdit.setText(f"✅ Model trained and saved! Test accuracy: {test_acc:.2f}")

            # Accuracy Graph
            plt.figure()
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('Accuracy.png')
            plt.close()

            # Loss Graph
            plt.figure()
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('Loss.png')
            plt.close()
            
        except Exception as e:
            self.textEdit.setText(f"❌ Error during training: {str(e)}")


# ===================== Run App =====================
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())