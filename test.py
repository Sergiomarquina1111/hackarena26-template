import face_recognition
img = face_recognition.load_image_file('known_faces/Prashant.jpg')
encs = face_recognition.face_encodings(img)
print('Faces found:', len(encs))
if encs:
    print('SUCCESS — Prashant.jpg is ready for recognition')
else:
    print('FAIL — No face detected in photo. Try a clearer front-facing photo.')