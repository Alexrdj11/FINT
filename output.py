import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import pickle
import warnings
from PIL import Image
import base64


warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="FINT",
                   layout="wide",
                   page_icon="apple")

# Function to set a background image from a URL
def set_bg_from_url(url,opacity=0.001):
    page_bg_img = f"""
    <style>
    .stApp {{
    background-image: url("{url}");
    background-size: cover;
    background-attachment: fixed;  # Ensure the image doesn't scroll with the page
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the URL of the background image
set_bg_from_url('https://c4.wallpaperflare.com/wallpaper/899/593/118/cuisine-food-india-indian-wallpaper-preview.jpg')


# Load the model
model = tf.keras.models.load_model('F:/FINT_Symbiot/final_model.h5')

# Application title and description
st.markdown("<h1 style='text-align: center; color: white;'>FINT -Your Food Image & Nutrition Tracker</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: white;'>Welcome to the FINT. Please upload your file below.</h5>", unsafe_allow_html=True)



# Define a function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Nutritional information dictionary
nutritional_info = {
   'adhirasam': {'calories': '434kcal', 'protein': '4.9g', 'fat': '18.4g','saturated fat':'3.3g','monounsaturated fat':'9.8g','polyunsaturated fat':'4.7g', 'carbs': '62.8g','sugars':'23.6g','fibers':'1.2g'}, 
    'aloo_gobi': {'calories': '89kcal', 'protein': '2.8g', 'fat': '3.7g','saturated fat':'0.6g','monounsaturated fat':'1.9g','polyunsaturated fat':'0.7g', 'carbs': '12.8g','sugars':'2.6g','fibers':'3.2g'},
    'aloo_matar': {'calories': '120kcal', 'protein': '3.9g', 'fat': '4.1g','saturated fat':'0.7g','monounsaturated fat':'2.8g','polyunsaturated fat':'0.7g', 'carbs': '18.8g','sugars':'3.6g','fibers':'4.2g'},
     'aloo_methi': {'calories': '105kcal', 'protein': '3.8g', 'fat': '3.4g','saturated fat':'0.6g','monounsaturated fat':'2.8g','polyunsaturated fat':'0.7g', 'carbs': '15.8g','sugars':'2.6g','fibers':'4.2g'},
      'aloo_shimla_mirch': {'calories': '95kcal', 'protein': '2.9g', 'fat': '4.4g','saturated fat':'0.6g','monounsaturated fat':'2.8g','polyunsaturated fat':'1.0g', 'carbs': '13.8g','sugars':'2.6g','fibers':'3.2g'},
       'aloo_tikki': {'calories': '210kcal', 'protein': '3.9g', 'fat': '12.4g','saturated fat':'1.3g','monounsaturated fat':'6.8g','polyunsaturated fat':'3.7g', 'carbs': '23.8g','sugars':'1.6g','fibers':'2.8g'},
        'aloo_tikki': {'calories': '210kcal', 'protein': '3.9g', 'fat': '12.4g','saturated fat':'1.3g','monounsaturated fat':'6.8g','polyunsaturated fat':'3.7g', 'carbs': '23.8g','sugars':'1.6g','fibers':'2.8g'},
        'anarsa': {'calories': '385kcal', 'protein': '3.8g', 'fat': '13.4g','saturated fat':'10.3g','monounsaturated fat':'1.9g','polyunsaturated fat':'0.7g', 'carbs': '63.8g','sugars':'30.6g','fibers':'1.7g'},
         'ariselu': {'calories': '398kcal', 'protein': '3.5g', 'fat': '8.4g','saturated fat':'5.1g','monounsaturated fat':'1.6g','polyunsaturated fat':'0.7g', 'carbs': '77.8g','sugars':'38.6g','fibers':'1.2g'},
         'bandar_ladoo': {'calories': '398kcal', 'protein': '3.5g', 'fat': '8.4g','saturated fat':'5.1g','monounsaturated fat':'1.8g','polyunsaturated fat':'0.7g', 'carbs': '77.8g','sugars':'38.6g','fibers':'1.2g'},
          'basundi': {'calories': '278kcal', 'protein': '6.9g', 'fat': '13.4g','saturated fat':'8.3g','monounsaturated fat':'3.8g','polyunsaturated fat':'0.7g', 'carbs': '62.8g','sugars':'28.6g','fibers':'0.1g'},
          
            'bhatura': {'calories': '347kcal', 'protein': '8.3g', 'fat': '14.4g','saturated fat':'2.3g','monounsaturated fat':'7.8g','polyunsaturated fat':'4.0g', 'carbs': '46.8g','sugars':'0.6g','fibers':'1.6g'},
            'bhindi_masala': {'calories': '96kcal', 'protein': '2.4g', 'fat': '5.4g','saturated fat':'0.8g','monounsaturated fat':'9.8g','polyunsaturated fat':'1.7g', 'carbs': '10.8g','sugars':'4.6g','fibers':'4.2g'},
           
            'biriyani': {'calories': '350kcal', 'protein': '15g', 'fat': '10g','saturated fat':'3g','monounsaturated fat':'5g','polyunsaturated fat':'4g', 'carbs': '55g','sugars':'3.6g','fibers':'2g'},
             'boondi': {'calories': '525kcal', 'protein': '10g', 'fat': '33g','saturated fat':'3g','monounsaturated fat':'15g','polyunsaturated fat':'12g', 'carbs': '45g','sugars':'4g','fibers':'7g'},
              'butter_chicken': {'calories': '525kcal', 'protein': '10g', 'fat': '33g','saturated fat':'3g','monounsaturated fat':'15g','polyunsaturated fat':'12g', 'carbs': '45g','sugars':'4g','fibers':'7g'},
            'cham_cham': {'calories': '150kcal', 'protein': '6g', 'fat': '5g','saturated fat':'3g','monounsaturated fat':'2g','polyunsaturated fat':'1g', 'carbs': '30g','sugars':'20g','fibers':'1g'},
            'channa_masala': {'calories': '250kcal', 'protein': '18g', 'fat': '9g','saturated fat':'2g','monounsaturated fat':'3g','polyunsaturated fat':'3g', 'carbs': '40g','sugars':'5g','fibers':'10g'},
            'chapati': {'calories': '120kcal', 'protein': '3g', 'fat': '2g','saturated fat':'0.4g','monounsaturated fat':'1g','polyunsaturated fat':'0.5g', 'carbs': '20g','sugars':'0.3g','fibers':'2g'},
            'chhenna_kheeri': {'calories': '250kcal', 'protein': '8g', 'fat': '15g','saturated fat':'5g','monounsaturated fat':'2g','polyunsaturated fat':'1g', 'carbs': '30g','sugars':'25g','fibers':'1g'},
            'chak_hoo_kheer': {'calories': '250kcal', 'protein': '8g', 'fat': '5g','saturated fat':'3g','monounsaturated fat':'2g','polyunsaturated fat':'1g', 'carbs': '40g','sugars':'20g','fibers':'1g'},
            'chicken_tikka_masala': {'calories': '250kcal', 'protein': '8g', 'fat': '15g','saturated fat':'5g','monounsaturated fat':'2g','polyunsaturated fat':'1g', 'carbs': '30g','sugars':'25g','fibers':'1g'},
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
          
          'dal_tadka': {'Name':'Dal Tadka', 'calories': ' 169kcal ', 'protein': ' 9.2g ', 'Total fat': '4.5g ', 'Saturated fat':'0.6g ', 
                'mono unsaturated fat':'2.1g ', 'Polyunsaturated fat':'1.4g ' ,
                'carbs': '23.4g ', 'Sugar':'2.7g', 'Fiber' :'7.6g '},
          
          'dharwad_pedha': {'Name':'Dharwad Pedha', 'calories': '  357 kcal ', 'protein': '6.8g ', 'Total fat': '13.9g ', 'Saturated fat':'6.1g ', 
                'mono unsaturated fat':'6.1g ', 'Polyunsaturated fat':'1.0g ' ,
                'carbs': '52.3g ', 'Sugar':'38.8g', 'Fiber' :'0.9g '},
          
            ' doodhpak': {'Name':'Doodhpak', 'calories': '395K', 'protein': '10.4g', 'Total fat': '17.4g', 'Saturated fat':'17.4g', 'mono unsaturated fat':'3.7g', 'Polyunsaturated fat':'2.5g' ,'carbs': '49.4g', 'Sugar':'27.9g', 'Fiber' :'9.8g'},
            
            'double_ka_meetha': {'Name':' Double Ka Meetha', 'calories': ' 410 Kcal ', 'protein': '11.1g ', 'Total fat': '14.9g ', 'Saturated fat':' 5.5g ', 
                'mono unsaturated fat':'  3.3g ', 'Polyunsaturated fat':'1.1g ' ,
                'carbs': ' 57.7g ', 'Sugar':' 42.7g ', 'Fiber' :' 1.3g '},
             
            ' dum_aloo ': {'Name':' Dum Aloo ', 'calories': ' 214kcal ', 'protein': ' 3.5g ', 'Total fat': ' 11.3g ', 'Saturated fat':' 1.6g ', 'mono unsaturated fat':' 6.5g ', 'Polyunsaturated fat':' 2.8g ' ,'carbs': ' 26.7g ', 'Sugar':'2.1g', 
            'Fiber' :' 3.4g '}, 
             
            ' gajar_ka_halwa': {'Name':' Gajar Ki Halwa ', 'calories': ' 313 kcal ', 'protein': ' 3.1g ', 'Total fat': ' 16.6g ', 'Saturated fat':' 9.3g ', 'mono unsaturated fat':' 4.4g ', 'Polyunsaturated fat':' 2.1g ' ,'carbs': ' 38.6g ', 'Sugar':' 28.4g ', 'Fiber' :' 2.2g '},
             
             'gavvalu ': {'Name':' Gavvalu ', 'calories': ' 384 kcal ', 'protein': ' 10.7g ', 'Total fat': ' 4.7g ', 'Saturated fat':' 1.4g ', 'mono unsaturated fat':' 1.1g', 'Polyunsaturated fat':' 1.9g ' ,'carbs': ' 73.3g ', 'Sugar':' 28.4g ', 'Fiber' :' 10.7g '},
              
            ' ghevar ': {'Name':' Ghevar ', 'calories': ' 392 kcal ', 'protein': ' 7.6g ', 'Total fat': ' 12.9g ', 'Saturated fat':' 4.2g ', 'mono unsaturated fat':' 5.7g ', 'Polyunsaturated fat':' 2.2g ' ,'carbs': ' 63.8g ', 'Sugar':' 27.8g ', 'Fiber' :' 1.8g '},
               
            ' gulab_jamun ': {'Name':' Gulab Jamun ', 'calories': ' 293 kcal ', 'protein': ' 4.5g ', 'Total fat': ' 7.9g ', 'Saturated fat':' 4.4g ', 
                'mono unsaturated fat':' 2.1g ', 'Polyunsaturated fat':' 0.9g ' ,
                'carbs': ' 53.1g ', 'Sugar':' 33.8g ', 'Fiber' :' 1.3g '},
            
             ' imarti ': {'Name':'Imarti ', 'calories': ' 408 kcal ', 'protein': ' 5.1g ', 'Total fat': ' 17.8g ', 'Saturated fat':'8.7g ', 
                'mono unsaturated fat':' 6.3g ', 'Polyunsaturated fat':' 2.2g ' ,
                'carbs': ' 58.8g ', 'Sugar':' 34.1g ', 'Fiber' :'1.2g '},
             
             ' jalebi ': {'Name':'Jalebi ', 'calories': ' 318 kcal ', 'protein': ' 4.3g ', 'Total fat': ' 12.8g ', 'Saturated fat':'2.2g ', 
                'mono unsaturated fat':'6.5g ', 'Polyunsaturated fat':'  3.5g ' ,
                'carbs': ' 48.1g ', 'Sugar':'27.5g', 'Fiber' :'0.8g '},
             
              ' kachori ': {'Name':'Kachori ', 'calories': ' 404 kcal ', 'protein': '7.8g ', 'Total fat': ' 17.9g ', 'Saturated fat':' 3.3g ', 
                'mono unsaturated fat':'10.1g ', 'Polyunsaturated fat':'  3.7g ' ,
                'carbs': ' 55.1g ', 'Sugar':' 2.6g ', 'Fiber' :' 3.4g '},
              
               ' kadai_panner ': {'Name':'kadai panner ', 'calories': ' 350-400 kcal ', 'protein': '14-16g ', 'Total fat': ' 25-30g ', 'Saturated fat':' 3.3g ', 
                'mono unsaturated fat':'10.1g ', 'Polyunsaturated fat':'  3.7g ' ,
                'carbs': ' 20-25g ', 'Sugar':' 2.6g ', 'Fiber' :' 2-3g '},
               
             ' kadhi_pokoda ': {'Name':'Kadhi Pokoda ', 'calories': ' 192 kcal ', 'protein': '6.8g', 'Total fat': ' 13.7g ', 'Saturated fat':' 1.7g ', 
                'mono unsaturated fat':'7.1g ', 'Polyunsaturated fat':'  4.1g ' ,
                'carbs': ' 12.3g ', 'Sugar':' 2.9g ', 'Fiber' :' 2.6g '},
             
             ' kajjikaya': {'Name':'Kajjikaya ', 'calories': ' 375 kcal ', 'protein': '5.1g', 'Total fat': ' 27.4g ', 'Saturated fat':'  3.5g ', 
             'mono unsaturated fat':'15.1g ', 'Polyunsaturated fat':' 7.6g ' ,
            'carbs': ' 32g ', 'Sugar':' 1.9g ', 'Fiber' :' 5.6g '},
               
             ' kakinada_khaja': {'Name':'Kakinada Khaja ', 'calories': ' 485 kcal ', 'protein': '7.4g', 'Total fat': ' 26.1g ', 'Saturated fat':'  5.7g ', 
            'mono unsaturated fat':' 14.5g ', 'Polyunsaturated fat':' 5.1g ' ,
            'carbs': ' 55.6g ', 'Sugar':' 24.1g ', 'Fiber' :' 1.9g '},
             
             ' kalakand': {'Name':'kalakand ', 'calories': ' 343 kcal ', 'protein': '8.4g', 'Total fat': ' 16.6g ', 'Saturated fat':'  10g ', 
            'mono unsaturated fat':' 4.5g ', 'Polyunsaturated fat':'0.8g ' ,
            'carbs': ' 40.7g ', 'Sugar':' 33.4g', 'Fiber' :' 1g '},
             
             ' karela_bhrata': {'Name':'Karela Bhrata', 'calories': ' 27 kcal ', 'protein': '1.1g', 'Total fat': ' 0.2g ', 'Saturated fat':'   0.04g ', 
            'mono unsaturated fat':' 0.03g ', 'Polyunsaturated fat':'0.1g' ,
            'carbs': ' 5.8g ', 'Sugar':' 3.8g', 'Fiber' :' 2.8g'},
             
              ' kofta': {'Name':'Kofta', 'calories': ' 267 kcal ', 'protein': '11.5g', 'Total fat': ' 17.4g ', 'Saturated fat':'   4.3g ', 
            'mono unsaturated fat':' 7.2g ', 'Polyunsaturated fat':'4.6g' ,
            'carbs': ' 16.8g ', 'Sugar':' 2.6g', 'Fiber' :' 2.6g'},
              
               'kuzhi_paniyaram': {'Name':'Kuzhi Paniyaram', 'calories': ' 224 kcal ', 'protein': '4.7g', 'Total fat': ' 10.8g ', 'Saturated fat':'  2.2g ', 
            'mono unsaturated fat':' 5.2g ', 'Polyunsaturated fat':'2.9g' ,
            'carbs': ' 27.8g ', 'Sugar':' 2.1g', 'Fiber' :'1.9g'},
               
                'lassi': {'Name':'Lassi', 'calories': ' 150 kcal', 'protein': '6g', 'Total fat': ' 6g ', 'Saturated fat':'4g ', 
            'mono unsaturated fat':' 1g ', 'Polyunsaturated fat':'0.2g' ,
            'carbs': ' 18g', 'Sugar':' 17g', 'Fiber' :'0g'},
                
                'ledikeni': {'Name':'Ledikeni', 'calories': ' 293 kcal', 'protein': '4.5g', 'Total fat': ' 7.9g', 'Saturated fat':'4.4g ', 
            'mono unsaturated fat':'2.1g ', 'Polyunsaturated fat':'0.9g' ,
            'carbs': ' 53.1g ', 'Sugar':'33.8g', 'Fiber' :'1.3g'},
                
                'litti_chokha': {'Name':'Litti Chokha', 'calories': ' 177.9kcal', 'protein': ' 7.3g', 'Total fat': '  9.8g', 'Saturated fat':' 1.6g ', 
            'mono unsaturated fat':' 2.6 ', 'Polyunsaturated fat':'1.4g' ,
            'carbs': ' 27.7g ', 'Sugar':'2.5g', 'Fiber' :'3.5g'},
                
                'layangcha': {'Name':'Layangcha', 'calories': ' 120 kcal', 'protein': ' 1.5 g', 'Total fat': '  4.5g', 'Saturated fat':' 1.2 g ', 
            'mono unsaturated fat':'  1.5g ', 'Polyunsaturated fat':'1.8 g' ,
            'carbs': ' 22g ', 'Sugar':'12g', 'Fiber' :'1.5g'},
                
                 'layangcha': {'Name':'Layangcha', 'calories': ' 120 kcal', 'protein': ' 1.5 g', 'Total fat': '  4.5g', 'Saturated fat':' 1.2 g ', 
            'mono unsaturated fat':'  1.5g ', 'Polyunsaturated fat':'1.8 g' ,
            'carbs': ' 22g ', 'Sugar':'12g', 'Fiber' :'1.5g'},
               
             ' maach_jhol': {'Name':' Maach Jhol', 'calories': ' 215 kcal', 'protein': ' 22g', 'Total fat': ' 12g', 'Saturated fat':'2.5g', 
            'mono unsaturated fat':'  4g ', 'Polyunsaturated fat':'4g' ,
            'carbs': ' 7g ', 'Sugar':'1g', 'Fiber' :'1g'},
             
             ' makki_di_roti': {'Name':' Makki Di Roti', 'calories': ' 145kcal', 'protein': ' 3g', 'Total fat': ' 1.5g', 'Saturated fat':'0.2g', 
            'mono unsaturated fat':'  0.3g ', 'Polyunsaturated fat':'0.3g' ,
            'carbs': ' 29g ', 'Sugar':'1g', 'Fiber' :'12g'},
             
               ' malapua': {'Name':' Malapua', 'calories': ' 270kcal', 'protein': ' 4g', 'Total fat': ' 12g', 'Saturated fat':'2g', 
            'mono unsaturated fat':' 6g ', 'Polyunsaturated fat':'3.5g' ,
            'carbs': ' 37g ', 'Sugar':'12g', 'Fiber' :'1g'},
               
                ' misi_roti': {'Name':' Misi Roti', 'calories': '185kcal', 'protein': '5g', 'Total fat': ' 3g', 'Saturated fat':'0.5g', 
            'mono unsaturated fat':' 0.8g ', 'Polyunsaturated fat':'1.2g' ,
            'carbs': ' 35g ', 'Sugar':'1g', 'Fiber' :'5g'},
               
               ' misti_doi': {'Name':' Misti Doi', 'calories': '136kcal', 'protein': '3.8g', 'Total fat': ' 2.8g',
            'carbs': ' 24.1g ', 'Sugar':'10.5g', 'Fiber' :'0g'},
               
               'modak': {'Name':' Modak', 'calories': '185kcal', 'protein': '2.9g', 'Total fat': ' 9.5g', 'Saturated fat':'1.6g', 
            'mono unsaturated fat':' 0.7g ', 'Polyunsaturated fat':'2.4g' ,
            'carbs': ' 27.4g ', 'Sugar':'6.7g', 'Fiber' :'9.0g'},
               
               'mysore_pak': {'Name':'Mysore Pak', 'calories': '501.6kcal', 'protein': '7g', 'Total fat': ' 29.7g', 'Saturated fat':'9.8g', 
            'mono unsaturated fat':' 9.7g ', 'Polyunsaturated fat':'6.9g' ,
            'carbs': ' 27.4g ', 'Sugar':'39.5g', 'Fiber' :'4.0g'},
               
               'navrattan_korma': {'Name':'Naan', 'calories': '375kcal', 'protein': '9.4g', 'Total fat': '26.5g', 'Saturated fat':'2.9g', 
            'mono unsaturated fat':' 0.9g ', 'Polyunsaturated fat':'0.4g' ,
            'carbs': ' 36.2g ', 'Sugar':'0.2g', 'Fiber' :'5.2g'},
               
                'palak_paneer': {'Name':'Palak Paneer', 'calories': '289.3kcal', 'protein': '15.3g', 'Total fat': '19.4g', 'Saturated fat':'3.5g', 
            'mono unsaturated fat':' 0.3g ', 'Polyunsaturated fat':'0.1g' ,
            'carbs': ' 15.7g ', 'Sugar':'0.5g', 'Fiber' :'7.6g'},
                
                 'paneer_butter_masala': {'Name':'Paneer Butter Masala', 'calories': '398.0kcal', 'protein': '15.3g', 'Total fat': '27.4g', 'Saturated fat':'2.4g', 
            'mono unsaturated fat':' 0.3g ', 'Polyunsaturated fat':'0.6g' ,
            'carbs': ' 28.4g ', 'Sugar':'1.9g', 'Fiber' :'4.2g'},
                 
                  'phirni': {'Name':'Phirni', 'calories': '290.9kcal', 'protein': '3.1g', 'Total fat': '9.3g', 'Saturated fat':'1.3g', 
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
                 'sandesh': {'calories': '323kcal', 'protein': '7.9g', 'fat': '13.4g','saturated fat':'8.1g','monounsaturated fat':'4.1g','polyunsaturated fat':'0.7g', 'carbs': '41.8g','sugars':'34.6g','fibers':'0.2g'},  
            'shankarpali': {'calories': '457kcal', 'protein': '7.3g', 'fat': '22.4g','saturated fat':'3.4g','monounsaturated fat':'13.2g','polyunsaturated fat':'4.9g', 'carbs': '56.6g','sugars':'4.1g','fibers':'2.7g'},  
            'sheer_korma': {'calories': '368kcal', 'protein': '6.2g', 'fat': '16.8g','saturated fat':'8.5g','monounsaturated fat':'5.1g','polyunsaturated fat':'2.3g', 'carbs': '47.1g','sugars':'37.5g','fibers':'2.1g'},
            'sheera': {'calories': '376kcal', 'protein': '5.2g', 'fat': '12.1g','saturated fat':'6.1g','monounsaturated fat':'3.5g','polyunsaturated fat':'1.1g', 'carbs': '62.8g','sugars':'24.6g','fibers':'1.6g'},
            'shrikand': {'calories': '265kcal', 'protein': '9.2g', 'fat': '13.1g','saturated fat':'8.3g','monounsaturated fat':'3.4g','polyunsaturated fat':'0.7g', 'carbs': '28.8g','sugars':'25.6g','fibers':'0.2g'},
            'sohan_halwa': {'calories': '445kcal', 'protein': '6.9g', 'fat': '1.4g','saturated fat':'6.9g','monounsaturated fat':'4.7g','polyunsaturated fat':'1.8g', 'carbs': '72.8g','sugars':'53.6g','fibers':'1.9g'},
            'sohan_papdi': {'calories': '478kcal', 'protein': '5.9g', 'fat': '19.4g','saturated fat':'6.3g','monounsaturated fat':'9.2g','polyunsaturated fat':'3.1g', 'carbs': '69.8g','sugars':'28.6g','fibers':'1.8g'},
            'sutar_feni': {'calories': '223kcal', 'protein': '1.9g', 'fat': '20.4g','saturated fat':'3.3g','monounsaturated fat':'9.8g','polyunsaturated fat':'4.7g', 'carbs': '62.8g','sugars':'28.6g','fibers':'1.2g'},
            'unni_appam': {'calories': '195kcal', 'protein': '3.9g', 'fat': '8.4g','saturated fat':'5.4g','monounsaturated fat':'0.8g','polyunsaturated fat':'0.4g', 'carbs': '30.8g','sugars':'2.6g','fibers':'0.6g'},
}

# Streamlit app
st.markdown("<h1 style='text-align: center; color: white;'>Indian Food Image Classifier with Nutritional Information</h1>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get the class labels
    class_labels = ['adhirasam','aloo_gobi','aloo_matar','aloo_methi','aloo_shimla_mirch','aloo_tikki','anarsa','ariselu','bandar_ladoo','basundi',
                        'bhatura','bhindi_masala','biriyani','boondi','butter_chicken','cham_cham','channa_masala','chapati','chhenna_kheeri','chak_hoo_kheer','chicken_tikka_masala', 'chikki','daal_baati_churma','daal_puri','daal_makhni','dal_tadka','dharwad_pedha','doodhpak', 'double_ka_meetha', 'dum_aloo', ' gajar_ki_halwa', 'gavvalu ', ' ghevar ', ' gulab_jamun ', ' imarti ', ' jalebi ', ' kachori ', ' kadai_panner ', ' kadhi_pokoda ', ' kajjikaya', 'kakinada_khaja',' kalakand', 'karela_bhrata', ' kwofta', 'kuzhi_paniyaram',  'lassi', 'ledikeni', 'litti_chokha', 'layangcha', ' maach_jhol',  ' makki_di_roti', ' malapua', 'misi_roti','misti_doi','modak','mysore_pak','naan','navrattan_korma','palak_paneer','paneer_putter_masala','phirni','pithe','poha','poornalu','pootharekulu','qubani_ka_meetha','rabri','ras_malai', 'rasgulla','rasgulla''sandesh','shankarpali','sheer_korma','sheera','shrikand','sohan_halwa','sohan_papdi','sutar_feni','unni_appam']
    # Get the predicted label
    predicted_label = class_labels[predicted_class]

    st.markdown(f"<h2 style='text-align: center; color: white;'>Prediction: {predicted_label}</h2>", unsafe_allow_html=True)



    # Display the nutritional information
    nutrition = nutritional_info.get(predicted_label, 'Nutritional information not available')
    st.write('Nutritional Information:')
    st.write(nutrition)
