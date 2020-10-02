# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:15:05 2020

@author: PC
"""

def split_on_uppercase(s, keep_contiguous=False):
    """

    Args:
        s (str): string
        keep_contiguous (bool): flag to indicate we want to 
                                keep contiguous uppercase chars together

    Returns:

    """

    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or 
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])

    return parts



str1 = 'QUERY293Aixo es una query normal i corrent, la caption del qual es tal.r0-198.pngr1-198.pngr2-198.png'
str2 ='query88.pngAixo es una query normal i corrent, la caption del qual es talAixo es una query normal i corrent, la caption del qual es tal.Aixo es una query normal i corrent, la caption del qual es tal.Aixo es una query normal i corrent, la caption del qual es talAixo es una query normal i corrent, la caption del qual es tal.'
str3 = 'QUERY293Aixo es una query normal i corrent, la caption del qual es talr0-198.pngr1-198.pngr2-198.png'

#Per les del tipus com la 1
parts = str1.split('QUERY')
capts = split_on_uppercase(parts[1])
print('QUERY',capts[0])
cappart = capts[1].split('r0')
print(cappart[0])
number = cappart[1].split('.')
number = number[0].split('-')

for i in range(0,10):
    string = 'r'+str(i)+'-'+number[1]+'.png'
    print(string)


#Per les del tipus com la 2 ja ho tindríem
parts = str2.split('png')
capts = split_on_uppercase(parts[1])

string = parts[0]+'png'
print(string)
for i in capts:
    print(i)
    
parts = str3.split('QUERY')
capts = split_on_uppercase(parts[1])
print('QUERY',capts[0])
cappart = capts[1].split('r0')
print(cappart[0])
number = cappart[1].split('.')
number = number[0].split('-')
for i in range(0,10):
    string = 'r'+str(i)+'-'+number[1]+'.png'
    print(string)
    
query = 'query'
if query == 'QUERY':
    print('hey')
else:
    print('no')

