import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

# Load known face encodings
face_database = {"Shrit": "./faces/shrit.jpg"}
known_face_encodings = []
known_face_names = []

for name, image_file in face_database.items():
    image = face_recognition.load_image_file(image_file).astype('uint8')
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

attendance_log = set()  # Use a set to prevent duplicate entries in one session

# Define CSV file for attendance
today_date = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today_date}.csv"

# Create a CSV file for today if it doesn't already exist
with open(attendance_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time"])  # Headers for the CSV

def mark_attendance(name):
    """Marks the attendance by writing the name and timestamp to a CSV file if not already logged."""
    if name not in attendance_log:
        attendance_log.add(name)  # Add name to session log
        now = datetime.now().strftime("%Y-%m-%d")
        with open(attendance_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, now])  # Write the name and date to the CSV
        print(f"Attendance marked for {name} on {now}")

def process_frame(frame):
    """Processes each video frame to detect faces and mark attendance."""
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)  # Mark attendance if recognized
        
        # Draw rectangles and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, [name]):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    process_frame_toggle = True
    
    while True:
        ret, frame = video_capture.read()
        if process_frame_toggle:
            frame = process_frame(frame)
        process_frame_toggle = not process_frame_toggle
        cv2.imshow("Attendance Marker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
