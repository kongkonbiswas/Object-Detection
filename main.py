import cv2
video = cv2.VideoCapture("object.mp4")

while True:
    ret, frame = video.read()
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.desrtoyAllWindows()
