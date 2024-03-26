# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:54:09 2022

@author: Hp
"""
#shading correction sy hum apni pic k background ko or fore ground ko alag alg kr sakty hain ......
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball


img = cv2.imread("saa.jpeg" ,0)

radius = 10

final_img , background = subtract_background_rolling_ball(img , radius , light_background = True)


clahe = cv2.createCLAHE(clipLimit=3 , tileGridSize=(8,8))
clahe_img = clahe.apply(final_img)


cv2.imshow("orignal image", img)
cv2.imshow("Background image",background)
cv2.imshow("After background Subtraction", final_img)
cv2.imshow("After clahe", clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


