import numpy as np
import pandas as pd
import cv2
import pywt
from PIL import Image
from io import BytesIO
import base64
import pickle
import os

with open("model/celebrity_prediction_model.pickle", "rb") as file:
	model=pickle.load(file)

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')
eye_model = os.path.join(cv2_base_dir, 'data', 'haarcascade_eye.xml')

haar_cascade = cv2.CascadeClassifier(haar_model)
eye_cascade = cv2.CascadeClassifier(eye_model)

def get_cropped_faces_with_2_eyes(original_image):
	gray_image=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	faces=haar_cascade.detectMultiScale(gray_image, 1.2, 5)
	faces_list=[]
	for x,y,w,h in faces:
		face_gray=gray_image[y:y+h, x:x+w]
		eyes=eye_cascade.detectMultiScale(face_gray)
		if len(eyes)>=2:
			face_img=original_image[y:y+h, x:x+w]
			faces_list.append(face_img)
	return faces_list

def image_wavelet_transform(image, mode='haar', level=1):
	image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageArray=np.float32(image_gray)/255
	coeff=pywt.wavedec2(imageArray, mode, level=level)
	coeff[0]*=0
	imageArray_wt=pywt.waverec2(coeff, mode)
	imageArray_wt=np.uint8(imageArray_wt*255)
	return imageArray_wt

def array_encoded_image(image_array):
	image=Image.fromarray(image_array)
	buffer=BytesIO()
	image.save(buffer, format='JPEG')
	encoded_data=base64.b64encode(buffer.getvalue()).decode('utf-8')
	encoded_image="data:image/png;base64,"+encoded_data
	return encoded_image

def process_uploaded_image(file):
	image_bytes=file.read()
	image_array=np.frombuffer(image_bytes, dtype=np.uint8)
	return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def preprocess_predict_image(file=None, image_path=None):
	if file:
		original_image=process_uploaded_image(file)
	elif image_path:
		original_image=cv2.imread(image_path)
    
	try:
		faces=get_cropped_faces_with_2_eyes(original_image)
	except Exception as e:
		# print(e)
		return "Invalid Image! Upload Another Image File"
	
	faces=get_cropped_faces_with_2_eyes(original_image)
	faces_img=[]
	faces_name=[]
	faces_proba=[]
	
	for face_img in faces:
		face_color=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
		faces_img.append(array_encoded_image(face_color))
        
		img_scaled=cv2.resize(face_img, (64, 64))
		img_wt=image_wavelet_transform(face_img, mode='db1', level=5)
		img_wt_scaled=cv2.resize(img_wt, (64, 64))
		combined_img=np.vstack((img_scaled.reshape(-1,1), img_wt_scaled.reshape(-1,1)))
		combined_img=combined_img.reshape(1,-1)
		combined_img=np.float32(combined_img)
		
		name=model.predict(combined_img)[0]
		faces_name.append(name)
		proba=model.predict_proba(combined_img).max()
		faces_proba.append(round(proba,3))
	
	return pd.DataFrame({"face_img": faces_img, "face_name": faces_name, "face_proba": faces_proba})

if __name__=="__main__":
	path="static/images/test images"
	image=path+'/'+os.listdir(path)[0]
	pred_celebrities=preprocess_predict_image(image_path=image)
	if pred_celebrities.empty:
		print("no face found")
	"""for i,face in pred_celebrities.iterrows():
		plt.subplot(len(pred_celebrities),1,i+1)
		plt.imshow(face["face_img"])
		plt.title(face["face_name"])
	plt.show()"""