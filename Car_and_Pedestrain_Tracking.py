import cv2


# img_file = 'Car Image.jpg'
video = cv2.VideoCapture('cars.mp4')

classifier_file = 'car_detector.xml'
pedestrain_classifier = 'pedestrains.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_classifier)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrain = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255,0,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    for (x, y, w, h) in pedestrain:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    cv2.imshow("JP Scriven Car & Pedestrain Tracker", frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

video.release()

"""
frame = cv2.imread(img_file)

car_tracker = cv2.CascadeClassifier(classifier_file)

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_tracker.detectMultiScale(black_n_white)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("JP Scriven Car & Pedestrain Tracker", img)

cv2.waitKey()

"""
print("Code Completed!!!")