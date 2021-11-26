import cv2
import numpy as np


WIDTH, HEIGHT = 10, 20
img_path = r"C:\Users\Test\Documents\School\2021-H\IDATT2502\Course Work\Project\Documentation\Screencaps\gifs\qqqq.png"
img = cv2.imread(img_path)

img = img[47:208, 95:176, 0] * 0.299 + img[47:208, 95:176, 1] * 0.587 + img[47:208, 95:176, 2] * 0.114
cv2.imshow("before_resize", img)  # For debugging image crop
cv2.waitKey(0)
cv2.destroyAllWindows()

resized_screen = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
cv2.imshow("after_resize", resized_screen)  # For debugging image rescaling
cv2.waitKey(0)
cv2.destroyAllWindows()

x_t = np.reshape(resized_screen, [HEIGHT, WIDTH, 1])
cv2.imshow("after_reshape", x_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
x_t = x_t.astype(np.uint8)

x_t = np.array(x_t).astype(np.float32) / 125
x_t = np.round(x_t)

print(x_t)
