
# coding: utf-8

# In[1]:


import re

def clean_text(x, mode):
    if mode == 'numbers':
        return ''.join(r for r in re.findall(r'[0-9]+', str(x)))
    elif mode == 'text':
        return ''.join(r for r in re.findall(r'[0-9]+', str(x).lower()))
    else:
        return 'choose correct mode value [numbers, text]'

