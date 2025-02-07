import cv2 as cv

img=cv.imread(r"C:\Users\Vishaal\OneDrive\Pictures\Camera Roll\WIN_20240526_10_33_38_Pro.jpg")

cv.imshow("Image",img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow("Grayed",gray)

haarCascade=cv.CascadeClassifier("haar_face.xml")

faces_rect=haarCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f"Number of faces founded={len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected Faces",img)

cv.waitKey(0)