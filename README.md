# Real-time-Facial-Expression-Recongnition
# This is code for the detection of Facial Expression in real-time video scenario.

import cv2
from deepface import DeepFace

""""
Deepface is a lightweight face recognition
and facial attribute analysis (age, gender, emotion and race)
framework for python
"""

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 

#haarcascade_frontalface_default.xml
"""
That is an XML file containing serialized Haar cascade detector of faces (Viola-Jones algorithm) 
in the OpenCV library. It is coded list of decision trees in which each vertex test one Haar Feature 
and each list claims “this is not face” or “this could be face”. It can be used the check that a part 
of image is face
"""

#To connect with webcam:
#start
cap=cv2.VideoCapture(1)

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
#End

    
while True:
    ret,frame=cap.read() #Reading image form video
    
    result=DeepFace.analyze(frame,actions=['emotion'])
    
    #Result variable:
    """
    Result variable will contain the dictionary in which key 
    of dictionary will be the emotions and values of keys are predicted percentage
    of emotions in the frames from image
    """
    
    #To draw rectangle across the face:
    #start
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    #End
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,
           result['dominant_emotion'], 
           (50,50),font,3,(0,0,255),2,cv2.LINE_4)#Print the Output of that percentage which have
                                                 #highest percentage in result variable
    
    cv2.imshow("Demo video",frame)
    
    if cv2.waitKey(2) & 0xFF== ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    
