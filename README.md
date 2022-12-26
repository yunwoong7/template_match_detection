# Template Match Detection
이미지 연산 방법을 통해 영역을 찾는 방법 중 네번째로 윤곽선이 아닌 **템플릿과 일치되는 영역을 검출하는 방법**에 대해 알아 보겠습니다. 사용하는 방법은 윤곽선 검출을 이용하는 것 만큼이나 간단합니다.

 

------

#### **Import packages**

```python
import numpy as np
from imutils.object_detection import non_max_suppression
import cv2
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

이미지에서 template과 일치되는 영역을 찾는 Function

```python
def template_matched_roi(img, template_img, threshold=0.9):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    (t_height, t_width) = template_img.shape[:2]
    
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    (yCoords, xCoords) = np.where(result >= threshold)
    
    roi_list = []
 
    for (x, y) in zip(xCoords, yCoords):
        roi_list.append((x, y, x + t_width, y + t_height))
    
    roi_array = non_max_suppression(np.array(roi_list))
    print("matched count : [{}]".format(len(roi_array)))
    
    return roi_array
```

#### **Load Image**

```python
cv2_image = cv2.imread('asset/images/cats.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/pRSDU/btrR8awVAjw/bauGonkwezQTooQRnNZWpk/img.png" width="50%">
</div>

template 이미지를 불러옵니다.

```python
template_image = cv2.imread('asset/images/cat_template.jpg', cv2.IMREAD_COLOR)
img_show('template image', template_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/EUzrE/btrR9krD7GG/QpgkSv8GWO8MppW5K3LvH0/img.png" width="50%">
</div>

#### **Template Match Detection**

```python
roi_array = template_matched_roi(cv2_image, template_image)
vis = cv2_image.copy()
 
for (x1, y1, x2, y2) in roi_array:
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
```

Template match를 표현한 이미지를 확인합니다.

```python
img_show(['matched image'], [vis])
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/biKh6V/btrR9unnZK2/j9husOxvLdtsJcOoj3WIEk/img.png" width="50%">
</div>

------

당연히 Template match는 완벽하지 않습니다. 템플릿과 동일한 영역의 이미지가 회전되었거나 크기가 많이 다른경우 실패합니다. 그렇지만 1줄 코드만으로도 수행이 가능하고, 연산 속도가 빠르며 윤곽선을 검출하는 방법처럼 임계값을 설정하지 않아도 됩니다.



작성하다가 어렸을 때 봤던 "월리를 찾아라!"가 생각나서 수행해 보았습니다. 이런 문제를 해결하는데 유용하겠네요.

```python
cv2_image = cv2.imread('asset/images/find_wally.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bRtTqN/btrR7JzUixX/AvbSevPGcNAAClxQzwwAi1/img.png" width="50%">
</div>

```python
template_image = cv2.imread('asset/images/wally.jpg', cv2.IMREAD_COLOR)
img_show('wally', [template_image], figsize=(2,1))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/dDAxHI/btrR7KFys1W/8Y12RuIqwIH4P3EDEVSHF1/img.png" width="50%">
</div>

```python
gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
 
result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
 
(startX, startY) = maxLoc
endX = startX + template_image.shape[1]
endY = startY + template_image.shape[0]
 
vis = cv2_image.copy()
vis = cv2.rectangle(vis, (startX, startY), (endX, endY), (0, 255, 0),3)
 
img_show('find wally', vis, figsize=(16, 10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bb20VK/btrR8aDHs88/XuX6UqMKk9xbr6WAfbDvz0/img.png" width="50%">
</div>
