import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import / pickle
 



# Load the trained food classification model
model = load_model('checkpoint/full_model.h5')

# Define class labels
class_labels = ['adhirasam','chikki','Doodhpak','daal_baati_churma','daal_puri','daal_makhni','dal_tadka','dharwad_pedha', 'Double Ka Meetha', 'Dum Aloo', ' Gajar Ki Halwa', 'Gavvalu ', ' Ghevar ', ' Gulab Jamun ', ' Imarti ', ' Jalebi ', ' Kachori ', ' kadai panner ', ' Kadhi Pokoda ', ' Kajjikaya',  'Kakinada_Khaja', ' kalakand', 'Karela Bhrata', ' Kwofta', 'Kuzhi Paniyaram',  'Lassi', 'ledikeni', 'Litti Chokha', 'Layangcha', ' Maach Jhol',  ' Makki Di Roti', ' Malapua', 'Misi Roti','Misti Doi','Modak','Mysore Pak','Naan','Navrattan Korma','Palak Paneer','Paneer Butter Masala','Phirni','pithe','poha','poornalu','pootharekulu','qubani_ka_meetha','rabri','ras_malai', 'rasgulla','rasgulla',
                        'sandesh','shankarpali','sheer_korma','sheera','shrikand','sohan_halwa','sohan_papdi','sutar_feni','unni_appam','allo_gobi','allo_matar','aloo_methi','aloo_shimla_mirch','aloo_tikki','anarsa','ariselu','bandar_ladoo','basundi',
                        'bhatura','bhindi_masala','boondi','biriyani','cham_cham','channa_masala','chapati','chhenna_kheeri','chak_hoo_kheer']  # List of your food class labels

# Function to preprocess image for model prediction
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to perform inference on a single frame
def predict_food(frame):
    # Preprocess the frame
    img = preprocess_image(frame)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Perform inference
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to open webcam")
    exit()

# Loop to continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame")
        break

    # Perform food prediction
    predicted_food = predict_food(frame)

    # Display the predicted food label on the frame
    cv2.putText(frame, predicted_food, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the captured frame
    cv2.imshow('Food Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
