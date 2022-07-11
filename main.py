import os
from datetime import datetime
from os import listdir
import cv2
import numpy as np
import face_recognition


class FaceCameraStreamingWidget:
    def __init__(self):
        #self.camera = cv2.VideoCapture(int(os.environ.get('CAMERA')))
        self.camera = cv2.VideoCapture(0)

    def load_images(self):
        image_list = []
        known_face_encodings = []
        known_face_names = []
        folder_dir = "images/"
        for images in os.listdir(folder_dir):
            image_list.append(images)
            known_face_names.append(images)
            image = face_recognition.load_image_file(f'{folder_dir}/{images}')
            image_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_encoding)
        print("Images loaded")
        return known_face_encodings, known_face_names
        

    def get_frames(self):
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        known_face_encodings, known_face_names = self.load_images()
        
        
        while True:
            # Capture frame-by-frame
            success, frame = self.camera.read()
            frame = np.array(frame)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
            process_this_frame = not process_this_frame
             # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                font = cv2.FONT_HERSHEY_DUPLEX
                if name in known_face_names:
                    cv2.rectangle(frame, (left, top - 100), (right, bottom), (0, 255, 0), 2)
                    #cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 2, bottom - 20), font, 1.0, (0, 255, 0), 1)
                # Draw a box around the face
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 50), (right, bottom), (5, 5, 300), cv2.FILLED)
                    cv2.putText(frame, name, (left + 2, bottom - 20), font, 1.0, (255, 255, 255), 1)
             
                    # Display the resulting image
            cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release handle to the webcam
        self.camera.release()
        cv2.destroyAllWindows()

                

start = FaceCameraStreamingWidget()
start.load_images()
start.get_frames()