import cv2

img = cv2.imread('python_V2hu73FElf.png')
cropped = img[47:209, 95:176, 0] * 0.299 + img[47:209, 95:176, 1] * 0.587 + img[47:209, 95:176, 2] * 0.114
resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)

