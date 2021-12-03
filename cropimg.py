import cv2

# print(img_crop.shape)
# cv2.waitKey(0)
from os import walk

filenames = next(walk('images/phuclong'), (None, None, []))[2]
print(filenames)
count = 0
for img in filenames:
    try:
        count = count + 1
        img = cv2.imread('images/phuclong/'+img)
        height, width, channels = img.shape
        print(height, width)
        img_crop = img[0:int(height / 2), 0:width]
        cv2.imwrite('output/phuclong/'+str(count)+ '.jpg', img_crop)
        cv2.waitKey(0)
    except:
        print("AAA")
