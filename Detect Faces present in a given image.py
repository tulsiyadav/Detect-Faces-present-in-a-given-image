#!/usr/bin/env python
# coding: utf-8

# # EDUNET FOUNDATION-Class Exercise Notebook

# ## LAB 1 - Detect faces present in a given image

# ### Load OpenCV library

# In[28]:


import cv2


# ### Load the cascade
# 

# In[29]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# ### Read the image

# In[30]:


img = cv2.imread('face.jpeg')


# ### Convert to grayscale
# 

# In[31]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ### Detect the faces
# 

# In[32]:


faces = face_cascade.detectMultiScale(gray, 1.1, 4)


# ### Draw the rectangle around each face
# 

# In[33]:


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# ### Show original image

# In[34]:


cv2.imshow('img', img)


# In[35]:


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




