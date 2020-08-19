import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('face_default.xml')
smile_detector = cv2.CascadeClassifier('smile.xml')

# Grab Webcam Feed
webcam = cv2.VideoCapture(0)  # 0 represents web cam

# Show the current frame
while True:
    # Read current frame from webcam stream
    successful_frame_read, frame = webcam.read()

    # If there's an error, abort
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and smiles
    faces = face_detector.detectMultiScale(frame_grayscale)
    #smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor=1.7, minNeighbors=20)

    # Run face detection whiting each of those frames
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)

        # Get the sub frame(using numpy N-dimensional array slicing
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Find all the smiles in the face
        # for (x_smiles, y_smiles, w_smiles, h_smiles) in smiles:

            # Draw a rectangle around the smile
            # cv2.rectangle(the_face, (x_smiles, y_smiles), (x_smiles + w_smiles, y_smiles + h_smiles), (50, 50, 200), 4)

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


    # Show the current frame
    cv2.imshow('Smile Detector', frame)

    # Display
    cv2.waitKey(1)  # automatically spams key every 1ms

# Cleanup
webcam.release()
cv2.destroyAllWindows()
