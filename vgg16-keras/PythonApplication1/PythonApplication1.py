import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
import csv
from scipy.spatial import distance
print(r''' 
  _____                                       _        _                 _           __   ____   
 |_   _|                                     | |      (_)               | |         /_ | |___ \  
   | |  _ __ ___   __ _  __ _  ___   _ __ ___| |_ _ __ _  _____   ____ _| | __   __  | |   __) | 
   | | | '_ ` _ \ / _` |/ _` |/ _ \ | '__/ _ \ __| '__| |/ _ \ \ / / _` | | \ \ / /  | |  |__ <  
  _| |_| | | | | | (_| | (_| |  __/ | | |  __/ |_| |  | |  __/\ V / (_| | |  \ V /   | |_ ___) | 
 |_____|_| |_| |_|\__,_|\__, |\___| |_|  \___|\__|_|  |_|\___| \_/ \__,_|_|   \_/    |_(_)____/  
                         __/ |                                                                   
                        |___/                                                                    
 
                              By Chrysostomos Pogiatzis                                                                                              
''')
model = VGG16(weights='imagenet')
model.summary()
lista_iconas=[]


for i in range(1,51):

   
      img_path = 'images/ucid'+str(i).zfill(5)+'.tif'
      
      img = image.load_img(img_path, target_size=(224, 224))
      img_data = image.img_to_array(img)
      img_data = np.expand_dims(img_data, axis=0)
      img_data = preprocess_input(img_data)

      vgg16_feature = model.predict(img_data)
      
      lista_iconas.append(vgg16_feature)
      print("The image "+ img_path +" it done")
your_query = 0
your_query = int(input('Enter the number of the image query:1-1337 \n'))
   
img_path = 'images/ucid'+str(i).zfill(5)+'.tif'
row=[distance.euclidean(lista_iconas[your_query-1],lista_iconas[i-1]),img_path]
with open(str(your_query)+'.csv', 'a') as csvFile:
     writer = csv.writer(csvFile)
     writer.writerow(row)

csvFile.close()


