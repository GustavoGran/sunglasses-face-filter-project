import cv2
import numpy as np


def get_overlayed_image(img, img_overlay, x, y):
	"""Overlay `img_overlay` onto `img` at (x,y)
	"""

	img_shape = img.shape[:2]
	bgr = img[:,:,0:3]
	if len(img) == 4:
		mask = img[:,:,3]
	else:
		mask = np.zeros(img_shape, dtype=np.uint8)
		mask = 255

	ov_height, ov_width = img_overlay.shape[:2]
	bgr_ov = img_overlay[:,:,0:3]
	mask_ov = img_overlay[:,:,3]

	new_bgr = bgr.copy()
	new_bgr[y:y+ov_height, x:x+ov_width] = bgr_ov

	new_mask = np.zeros(img_shape, dtype=np.uint8)
	new_mask[y:y+ov_height, x:x+ov_width] = mask_ov

	# combine two masks by bitwise and operation (multiply)
	combined_masks = cv2.multiply(mask, new_mask)
	combined_masks = cv2.cvtColor(combined_masks, cv2.COLOR_GRAY2BGR)

	# overlay base new_bgr into bgr using mask
	result = np.where(combined_masks == 255, new_bgr, bgr)

	return result

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,420)

# impor cascade file for facial recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# let's also import eyes detection harr classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# loads sunglasses
sunglasses = cv2.imread('./resources/meme-sunglasses.png', cv2.IMREAD_UNCHANGED)

while True:
	success, img = cap.read(cv2.IMREAD_UNCHANGED)
	# img = cv2.imread('./resources/test_face.jpg', cv2.IMREAD_UNCHANGED)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Getting corners around the face
	faces = face_cascade.detectMultiScale(img_gray, 1.3, 5) # 1.3 scale factor, 5 minimum neighbor

	img_bounding = img.copy()
	# drawing bounding box around face
	for (x,y,w,h) in faces:
		img_bounding = cv2.rectangle(img_bounding, (x,y), (x+w, y+h), (0,255,0), 3)

	eyes = eye_cascade.detectMultiScale(img_gray)

	eye_coords = [{'x':x,'y':y,'w':w,'h':h} for (x, y ,w, h) in eyes]

	# drawing bounding box for eyes
	for (x, y ,w, h) in eyes:
		img_bounding = cv2.rectangle(img_bounding, (x,y), (x+w, y+h), (255,0,0),3)

	if (len(eye_coords) == 2):
		width_glasses = int(eye_coords[1]['x']  - eye_coords[0]['x'] + eye_coords[1]['w']  + eye_coords[0]['w'])
		height_glasses = int((sunglasses.shape[0]/sunglasses.shape[1]) * width_glasses)
		resized_glasses = cv2.resize(sunglasses, (width_glasses, height_glasses))

		glass_x_center =  int(0.5*(eye_coords[0]['x'] + eye_coords[0]['w']) + 0.5*eye_coords[1]['x'])
		glass_y_center =  int(eye_coords[0]['y'] + 0.5*eye_coords[0]['w'])

		img_with_glasses = get_overlayed_image(
			img.copy(), resized_glasses, 
			glass_x_center - width_glasses // 2, 
			glass_y_center - height_glasses // 2
		)

	cv2.imshow('glasses', img_with_glasses)
	cv2.imshow('face_detect', img_bounding)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyWindow('face_detect')
cv2.destroyWindow('glasses')