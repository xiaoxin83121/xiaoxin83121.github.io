# CS131 Notebook

> A individual notebook for  cs131: computer vision

## Pixels and Filters

####  Color
Color: **The result of interaction between physical light in the environment and our visual system. **   
It's a **psychological propety** of our visual experiences, **NOT** physical property.  

#### Color Space
- RGB Cubic color space

![image-20200911234439191](https://i.loli.net/2020/09/11/GbpjIvt2zkoO1iE.png)

- HSV(Hue, Saturation, Value: 色调，饱和度，明度): cone

![image-20200911234539034](https://i.loli.net/2020/09/11/5kYp39vGgIcbPE8.png)  
``` python
img = cv2.imread("./test.jpg") # BGR==0:b, 1:g, 2:r
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_b = img[..., 0]
img_g = img[..., 1]
img_r = img[..., 2]
cv2.imshow("Image_B", img_b)
cv2.imshow("Image_G", img_g)
cv2.imshow("Image_R", img_r)
cv2.imshow("Image_BGR", img)
```

