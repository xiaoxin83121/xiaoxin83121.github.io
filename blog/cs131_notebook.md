# CS131 Notebook

> A individual notebook for  cs131: computer vision

## Pixels and Filters

###  Color
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

### Image sampling and quantization

- Resolution: sampling parameter, defined in dots per inch(DPI) or spatial pixel density.   
- An image contains discrete number of pixels: a matrix or a set of matrix(r,g,b,etc.)  

####  Image histograms

provide the frequency of the brightness(intensity) value. 

``` python
img = cv2.imread('./test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = np.zeros(255)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            h[img[row, col]] += 1
    plt.plot(h)
    plt.show()
```

![image-20200912201438185](https://i.loli.net/2020/09/12/X9exVM8E1LAQGJ7.png)  

### Linear systems(filter)
Filtering: Forming a new image whose pixel values are transformed from original pixel values. (两个矩阵之间的线性变换)  
详细内容可以查看信号与系统中关于Linear shift invariant system的详细描述  
Convolution手动实现：
``` python
def convolution_filter(stride=1):
    # manually convolution and see how it changes
    img = cv2.imread('./test_high.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # generate kernel
    kernel = np.ones((3, 3)) / 9
    # kernel = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]) / 9
    # generate output feature map
    output = np.zeros_like(img)
    time_before = time.time()
    for row in range(0, img.shape[0]-2, stride):
        for col in range(0, img.shape[1]-2, stride):
            mat = img[row:row+3, col:col+3] * kernel
            val = np.sum(mat.flatten())
            try:
                output[int(row/stride) + 1][int(col/stride) + 1] = val
            except:
                print('error happened when row={},col={}'.format(row, col))

    time_after = time.time()
    print("time consume={}".format(time_after- time_before))

    cv2.imshow("origin", img)
    cv2.imshow("convolution", output)
    cv2.waitKey()
    cv2.destroyAllWindows()
```

![image-20200912214818283](https://i.loli.net/2020/09/12/lImpdwJZsxY8A3U.png)

右图变糊了一点，实际测试[[0,1,2],[0,1,2],[0,1,2]]/9的卷积核差距不大，stride的调整也不会对整体观感有明显影响， origin file和convolution的结果之间的difference的结果如下：

<img src="https://i.loli.net/2020/09/12/bMu5oQPBG3SACc2.png" alt="image-20200912220536422" style="zoom:50%;" />