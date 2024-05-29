from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = tf.keras.models.load_model('path_to_your_final_model.h5')

# Define a function to load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # You should have a mapping from class indices to class names
        class_names = ['chikki','Doodhpak','daal_baati_churma','daal_puri','daal_makhni','dal_tadka','dharwad_pedha', 'Double Ka Meetha', 'Dum Aloo', ' Gajar Ki Halwa', 'Gavvalu ', ' Ghevar ', ' Gulab Jamun ', ' Imarti ', ' Jalebi ', ' Kachori ', ' kadai panner ', ' Kadhi Pokoda ', ' Kajjikaya',  'Kakinada Khaja', ' kalakand', 'Karela Bhrata', ' Kwofta', 'Kuzhi Paniyaram',  'Lassi', 'ledikeni', 'Litti Chokha', 'Layangcha', ' Maach Jhol',  ' Makki Di Roti', ' Malapua', 'Misi Roti','Misti Doi','Modak','Mysore Pak','Naan','Navrattan Korma','Palak Paneer','Paneer Butter Masala','Phirni','pithe','poha','poornalu','pootharekulu','qubani_ka_meetha','rabri','ras_malai', 'rasgulla','rasgulla',] 
        
        # Fill this list with your class names
        predicted_label = class_names[predicted_class[0]]

        # Map the predicted label to nutritional information
        nutritional_info = {
          #  'Class 1': {'calories': 100, 'protein': 10, 'fat': 5, 'carbs': 20}, 
          
          'chikki': {'Name':'Chikki ', 'calories': ' 318 kcal ', 'protein': ' 6g ', 'Total fat': ' 27g ', 'Saturated fat':'2.2g ', 
                'mono unsaturated fat':'8g ', 'Polyunsaturated fat':' 2g ' ,
                'carbs': '62g ', 'Sugar':'38g', 'Fiber' :'1g '},
          
          'daal_baati_churma': {'Name':'Daal Baati Churma ', 'calories': '412 kcal ', 'protein': '13.2g ', 'Total fat': '11.5g ', 'Saturated fat':'2.6g ', 
                'mono unsaturated fat':'1.4g', 'Polyunsaturated fat':'2.9g ' ,
                'carbs': '63.9g  ', 'Sugar':'3.2g', 'Fiber' :'8.3g '},
          
          'daal_puri': {'Name':'Daal Puri ', 'calories': ' 334 kcal ', 'protein': '8.2g ', 'Total fat': '14.5g ', 'Saturated fat':'2.1g ', 
                'mono unsaturated fat':'6.3g ', 'Polyunsaturated fat':'5.5g ' ,
                'carbs': ' 44.2g ', 'Sugar':'1.1g', 'Fiber' :'5.1g '},
          
          'daal_makhni': {'Name':'Daal Makhni ', 'calories': ' 227kcal ', 'protein': '9.1g ', 'Total fat': ' 14.6g ', 'Saturated fat':' 8.5g ', 
                'mono unsaturated fat':'4.2g ', 'Polyunsaturated fat':'1.2g ' ,
                'carbs': '16.7g ', 'Sugar':'2.9g', 'Fiber' :'5.2g '},
          
          'dal_tadka': {'Name':'DAl Tadka', 'calories': ' 169kcal ', 'protein': ' 9.2g ', 'Total fat': '4.5g ', 'Saturated fat':'0.6g ', 
                'mono unsaturated fat':'2.1g ', 'Polyunsaturated fat':'1.4g ' ,
                'carbs': '23.4g ', 'Sugar':'2.7g', 'Fiber' :'7.6g '},
          
          'dharwad_pedha': {'Name':'Dharwad Pedha', 'calories': '  357 kcal ', 'protein': '6.8g ', 'Total fat': '13.9g ', 'Saturated fat':'6.1g ', 
                'mono unsaturated fat':'6.1g ', 'Polyunsaturated fat':'1.0g ' ,
                'carbs': '52.3g ', 'Sugar':'38.8g', 'Fiber' :'0.9g '},
          
            ' Doodhpak': {'Name':'Doodhpak', 'calories': '395K', 'protein': '10.4g', 'Total fat': '17.4g', 'Saturated fat':'17.4g', 'mono unsaturated fat':'3.7g', 'Polyunsaturated fat':'2.5g' ,'carbs': '49.4g', 'Sugar':'27.9g', 'Fiber' :'9.8g'},
            
            'Double Ka Meetha': {'Name':' Double Ka Meetha', 'calories': ' 410 Kcal ', 'protein': '11.1g ', 'Total fat': '14.9g ', 'Saturated fat':' 5.5g ', 
                'mono unsaturated fat':'  3.3g ', 'Polyunsaturated fat':'1.1g ' ,
                'carbs': ' 57.7g ', 'Sugar':' 42.7g ', 'Fiber' :' 1.3g '},
             
            ' Dum Aloo ': {'Name':' Dum Aloo ', 'calories': ' 214kcal ', 'protein': ' 3.5g ', 'Total fat': ' 11.3g ', 'Saturated fat':' 1.6g ', 'mono unsaturated fat':' 6.5g ', 'Polyunsaturated fat':' 2.8g ' ,'carbs': ' 26.7g ', 'Sugar':'2.1g', 
            'Fiber' :' 3.4g '}, 
             
            ' Gajar Ki Halwa': {'Name':' Gajar Ki Halwa ', 'calories': ' 313 kcal ', 'protein': ' 3.1g ', 'Total fat': ' 16.6g ', 'Saturated fat':' 9.3g ', 'mono unsaturated fat':' 4.4g ', 'Polyunsaturated fat':' 2.1g ' ,'carbs': ' 38.6g ', 'Sugar':' 28.4g ', 'Fiber' :' 2.2g '},
             
             'Gavvalu ': {'Name':' Gavvalu ', 'calories': ' 384 kcal ', 'protein': ' 10.7g ', 'Total fat': ' 4.7g ', 'Saturated fat':' 1.4g ', 'mono unsaturated fat':' 1.1g', 'Polyunsaturated fat':' 1.9g ' ,'carbs': ' 73.3g ', 'Sugar':' 28.4g ', 'Fiber' :' 10.7g '},
              
            ' Ghevar ': {'Name':' Ghevar ', 'calories': ' 392 kcal ', 'protein': ' 7.6g ', 'Total fat': ' 12.9g ', 'Saturated fat':' 4.2g ', 'mono unsaturated fat':' 5.7g ', 'Polyunsaturated fat':' 2.2g ' ,'carbs': ' 63.8g ', 'Sugar':' 27.8g ', 'Fiber' :' 1.8g '},
               
            ' Gulab Jamun ': {'Name':' Gulab Jamun ', 'calories': ' 293 kcal ', 'protein': ' 4.5g ', 'Total fat': ' 7.9g ', 'Saturated fat':' 4.4g ', 
                'mono unsaturated fat':' 2.1g ', 'Polyunsaturated fat':' 0.9g ' ,
                'carbs': ' 53.1g ', 'Sugar':' 33.8g ', 'Fiber' :' 1.3g '},
            
             ' Imarti ': {'Name':'Imarti ', 'calories': ' 408 kcal ', 'protein': ' 5.1g ', 'Total fat': ' 17.8g ', 'Saturated fat':'8.7g ', 
                'mono unsaturated fat':' 6.3g ', 'Polyunsaturated fat':' 2.2g ' ,
                'carbs': ' 58.8g ', 'Sugar':' 34.1g ', 'Fiber' :'1.2g '},
             
             ' Jalebi ': {'Name':'Jalebi ', 'calories': ' 318 kcal ', 'protein': ' 4.3g ', 'Total fat': ' 12.8g ', 'Saturated fat':'2.2g ', 
                'mono unsaturated fat':'6.5g ', 'Polyunsaturated fat':'  3.5g ' ,
                'carbs': ' 48.1g ', 'Sugar':'27.5g', 'Fiber' :'0.8g '},
             
              ' Kachori ': {'Name':'Kachori ', 'calories': ' 404 kcal ', 'protein': '7.8g ', 'Total fat': ' 17.9g ', 'Saturated fat':' 3.3g ', 
                'mono unsaturated fat':'10.1g ', 'Polyunsaturated fat':'  3.7g ' ,
                'carbs': ' 55.1g ', 'Sugar':' 2.6g ', 'Fiber' :' 3.4g '},
              
               ' kadai panner ': {'Name':'kadai panner ', 'calories': ' 350-400 kcal ', 'protein': '14-16g ', 'Total fat': ' 25-30g ', 'Saturated fat':' 3.3g ', 
                'mono unsaturated fat':'10.1g ', 'Polyunsaturated fat':'  3.7g ' ,
                'carbs': ' 20-25g ', 'Sugar':' 2.6g ', 'Fiber' :' 2-3g '},
               
             ' Kadhi Pokoda ': {'Name':'Kadhi Pokoda ', 'calories': ' 192 kcal ', 'protein': '6.8g', 'Total fat': ' 13.7g ', 'Saturated fat':' 1.7g ', 
                'mono unsaturated fat':'7.1g ', 'Polyunsaturated fat':'  4.1g ' ,
                'carbs': ' 12.3g ', 'Sugar':' 2.9g ', 'Fiber' :' 2.6g '},
             
             ' Kajjikaya': {'Name':'Kajjikaya ', 'calories': ' 375 kcal ', 'protein': '5.1g', 'Total fat': ' 27.4g ', 'Saturated fat':'  3.5g ', 
             'mono unsaturated fat':'15.1g ', 'Polyunsaturated fat':' 7.6g ' ,
            'carbs': ' 32g ', 'Sugar':' 1.9g ', 'Fiber' :' 5.6g '},
               
             ' Kakinada Khaja': {'Name':'Kakinada Khaja ', 'calories': ' 485 kcal ', 'protein': '7.4g', 'Total fat': ' 26.1g ', 'Saturated fat':'  5.7g ', 
            'mono unsaturated fat':' 14.5g ', 'Polyunsaturated fat':' 5.1g ' ,
            'carbs': ' 55.6g ', 'Sugar':' 24.1g ', 'Fiber' :' 1.9g '},
             
             ' kalakand': {'Name':'kalakand ', 'calories': ' 343 kcal ', 'protein': '8.4g', 'Total fat': ' 16.6g ', 'Saturated fat':'  10g ', 
            'mono unsaturated fat':' 4.5g ', 'Polyunsaturated fat':'0.8g ' ,
            'carbs': ' 40.7g ', 'Sugar':' 33.4g', 'Fiber' :' 1g '},
             
             ' Karela Bhrata': {'Name':'Karela Bhrata', 'calories': ' 27 kcal ', 'protein': '1.1g', 'Total fat': ' 0.2g ', 'Saturated fat':'   0.04g ', 
            'mono unsaturated fat':' 0.03g ', 'Polyunsaturated fat':'0.1g' ,
            'carbs': ' 5.8g ', 'Sugar':' 3.8g', 'Fiber' :' 2.8g'},
             
              ' Kofta': {'Name':'Kofta', 'calories': ' 267 kcal ', 'protein': '11.5g', 'Total fat': ' 17.4g ', 'Saturated fat':'   4.3g ', 
            'mono unsaturated fat':' 7.2g ', 'Polyunsaturated fat':'4.6g' ,
            'carbs': ' 16.8g ', 'Sugar':' 2.6g', 'Fiber' :' 2.6g'},
              
               'Kuzhi Paniyaram': {'Name':'Kuzhi Paniyaram', 'calories': ' 224 kcal ', 'protein': '4.7g', 'Total fat': ' 10.8g ', 'Saturated fat':'  2.2g ', 
            'mono unsaturated fat':' 5.2g ', 'Polyunsaturated fat':'2.9g' ,
            'carbs': ' 27.8g ', 'Sugar':' 2.1g', 'Fiber' :'1.9g'},
               
                'Lassi': {'Name':'Lassi', 'calories': ' 150 kcal', 'protein': '6g', 'Total fat': ' 6g ', 'Saturated fat':'4g ', 
            'mono unsaturated fat':' 1g ', 'Polyunsaturated fat':'0.2g' ,
            'carbs': ' 18g', 'Sugar':' 17g', 'Fiber' :'0g'},
                
                'ledikeni': {'Name':'Ledikeni', 'calories': ' 293 kcal', 'protein': '4.5g', 'Total fat': ' 7.9g', 'Saturated fat':'4.4g ', 
            'mono unsaturated fat':'2.1g ', 'Polyunsaturated fat':'0.9g' ,
            'carbs': ' 53.1g ', 'Sugar':'33.8g', 'Fiber' :'1.3g'},
                
                'Litti Chokha': {'Name':'Litti Chokha', 'calories': ' 177.9kcal', 'protein': ' 7.3g', 'Total fat': '  9.8g', 'Saturated fat':' 1.6g ', 
            'mono unsaturated fat':' 2.6 ', 'Polyunsaturated fat':'1.4g' ,
            'carbs': ' 27.7g ', 'Sugar':'2.5g', 'Fiber' :'3.5g'},
                
                'Layangcha': {'Name':'Layangcha', 'calories': ' 120 kcal', 'protein': ' 1.5 g', 'Total fat': '  4.5g', 'Saturated fat':' 1.2 g ', 
            'mono unsaturated fat':'  1.5g ', 'Polyunsaturated fat':'1.8 g' ,
            'carbs': ' 22g ', 'Sugar':'12g', 'Fiber' :'1.5g'},
                
                 'Layangcha': {'Name':'Layangcha', 'calories': ' 120 kcal', 'protein': ' 1.5 g', 'Total fat': '  4.5g', 'Saturated fat':' 1.2 g ', 
            'mono unsaturated fat':'  1.5g ', 'Polyunsaturated fat':'1.8 g' ,
            'carbs': ' 22g ', 'Sugar':'12g', 'Fiber' :'1.5g'},
               
             ' Maach Jhol': {'Name':' Maach Jhol', 'calories': ' 215 kcal', 'protein': ' 22g', 'Total fat': ' 12g', 'Saturated fat':'2.5g', 
            'mono unsaturated fat':'  4g ', 'Polyunsaturated fat':'4g' ,
            'carbs': ' 7g ', 'Sugar':'1g', 'Fiber' :'1g'},
             
             ' Makki Di Roti': {'Name':' Makki Di Roti', 'calories': ' 145kcal', 'protein': ' 3g', 'Total fat': ' 1.5g', 'Saturated fat':'0.2g', 
            'mono unsaturated fat':'  0.3g ', 'Polyunsaturated fat':'0.3g' ,
            'carbs': ' 29g ', 'Sugar':'1g', 'Fiber' :'12g'},
             
               ' Malapua': {'Name':' Malapua', 'calories': ' 270kcal', 'protein': ' 4g', 'Total fat': ' 12g', 'Saturated fat':'2g', 
            'mono unsaturated fat':' 6g ', 'Polyunsaturated fat':'3.5g' ,
            'carbs': ' 37g ', 'Sugar':'12g', 'Fiber' :'1g'},
               
                ' Misi Roti': {'Name':' Misi Roti', 'calories': '185kcal', 'protein': '5g', 'Total fat': ' 3g', 'Saturated fat':'0.5g', 
            'mono unsaturated fat':' 0.8g ', 'Polyunsaturated fat':'1.2g' ,
            'carbs': ' 35g ', 'Sugar':'1g', 'Fiber' :'5g'},
               
               ' Misti Doi': {'Name':' Misti Doi', 'calories': '136kcal', 'protein': '3.8g', 'Total fat': ' 2.8g',
            'carbs': ' 24.1g ', 'Sugar':'10.5g', 'Fiber' :'0g'},
               
               'Modak': {'Name':' Modak', 'calories': '185kcal', 'protein': '2.9g', 'Total fat': ' 9.5g', 'Saturated fat':'1.6g', 
            'mono unsaturated fat':' 0.7g ', 'Polyunsaturated fat':'2.4g' ,
            'carbs': ' 27.4g ', 'Sugar':'6.7g', 'Fiber' :'9.0g'},
               
               'Mysore Pak': {'Name':'Mysore Pak', 'calories': '501.6kcal', 'protein': '7g', 'Total fat': ' 29.7g', 'Saturated fat':'9.8g', 
            'mono unsaturated fat':' 9.7g ', 'Polyunsaturated fat':'6.9g' ,
            'carbs': ' 27.4g ', 'Sugar':'39.5g', 'Fiber' :'4.0g'},
               
               'Navrattan Korma': {'Name':'Naan', 'calories': '375kcal', 'protein': '9.4g', 'Total fat': '26.5g', 'Saturated fat':'2.9g', 
            'mono unsaturated fat':' 0.9g ', 'Polyunsaturated fat':'0.4g' ,
            'carbs': ' 36.2g ', 'Sugar':'0.2g', 'Fiber' :'5.2g'},
               
                'Palak Paneer': {'Name':'Palak Paneer', 'calories': '289.3kcal', 'protein': '15.3g', 'Total fat': '19.4g', 'Saturated fat':'3.5g', 
            'mono unsaturated fat':' 0.3g ', 'Polyunsaturated fat':'0.1g' ,
            'carbs': ' 15.7g ', 'Sugar':'0.5g', 'Fiber' :'7.6g'},
                
                 'Paneer Butter Masala': {'Name':'Paneer Butter Masala', 'calories': '398.0kcal', 'protein': '15.3g', 'Total fat': '27.4g', 'Saturated fat':'2.4g', 
            'mono unsaturated fat':' 0.3g ', 'Polyunsaturated fat':'0.6g' ,
            'carbs': ' 28.4g ', 'Sugar':'1.9g', 'Fiber' :'4.2g'},
                 
                  'Phirni': {'Name':'Phirni', 'calories': '290.9kcal', 'protein': '3.1g', 'Total fat': '9.3g', 'Saturated fat':'1.3g', 
            'mono unsaturated fat':' 0.6g ', 'Polyunsaturated fat':'0.1g' ,
            'carbs': ' 52.8g ', 'Sugar':'36.8g', 'Fiber' :'1.9g'},
               
               
                'pithe': {'Name':'Pithe', 'calories': '280.9kcal', 'protein': '3.4g', 'Total fat': '9.6g', 'Saturated fat':'0.3g', 
            'mono unsaturated fat':' 0.4g ', 'Polyunsaturated fat':'0.6g' ,
            'carbs': ' 46.7g ', 'Sugar':'5.6g', 'Fiber' :'1.2g'},
                
                 'poha': {'Name':'Poha', 'calories': '218kcal', 'protein': '3.2g', 'Total fat': '0.5g', 'Saturated fat':'0.1g', 
            'mono unsaturated fat':' 0.0g ', 'Polyunsaturated fat':'0.9g' ,
            'carbs': ' 46.8g ', 'Sugar':'0.6g', 'Fiber' :'1.8g'},
                 
                'poornalu': {'Name':'Poornalu', 'calories': '496.7kcal', 'protein': '6.2g', 'Total fat': '20.5g', 'Saturated fat':'5.4g', 
            'mono unsaturated fat':'5.6 g ', 'Polyunsaturated fat':'1.4g' ,
            'carbs': ' 67.3g ', 'Sugar':'1.4g', 'Fiber' :'2.4g'},
                
                'pootharekulu': {'Name':'Pootharekulu', 'calories': '388.4kcal', 'protein': '3.2g', 'Total fat': '13.4g', 'Saturated fat':'5.6g', 
            'mono unsaturated fat':'2.3g ', 'Polyunsaturated fat':'1.5g' ,
            'carbs': '66.7g ', 'Sugar':'2g', 'Fiber' :'3.4g'},
                
                'qubani_ka_meetha': {'Name':'Qubani Ka Meetha', 'calories': '240kcal', 'protein': '1.5g', 'Total fat': ' 0.5g', 'Saturated fat':'0.1g', 
            'mono unsaturated fat':' 0.0g ', 'Polyunsaturated fat':'0.3g' ,
            'carbs': '59g ', 'Sugar':'36.4g', 'Fiber' :'2g'},
                
                'rabri': {'Name':'Rabri', 'calories': '480kcal', 'protein': '10g', 'Total fat': '25g', 'Saturated fat':'6.7g', 
            'mono unsaturated fat':' 2.3g ', 'Polyunsaturated fat':'4g' ,
            'carbs': '52g ', 'Sugar':'30g', 'Fiber' :'1g'},
                
                'ras_malai': {'Name':'Ras Malai', 'calories': '380.2kcal', 'protein': '9.9g', 'Total fat': '11.3g', 'Saturated fat':'2.3g', 
            'mono unsaturated fat':' 0.3g ', 'Polyunsaturated fat':'0.9g' ,
            'carbs': '55.6g ', 'Sugar':'40.6g', 'Fiber' :'1g'},
                
                'rasgulla': {'Name':'Rasgulla', 'calories': '160kcal', 'protein': '2g', 'Total fat': '4g', 'Saturated fat':'1.4g', 
            'mono unsaturated fat':'0.0g ', 'Polyunsaturated fat':'0.0g' ,
            'carbs': '29g ', 'Sugar':'31.1g', 'Fiber' :'0g'},
                
                 

                
             
               
               
              
             
             
            
             
             
             
             
             
             
             
            
            
            
            # Example data
            # Add all your class nutritional info here
        }
        nutrition = nutritional_info.get(predicted_label, 'Nutritional info not available')

        return jsonify({
            'predicted_label': predicted_label,
            'nutritional_info': nutrition
        })

if __name__ == '__main__':
    app.run(debug=True)
