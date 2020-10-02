# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:54:22 2020

@author: PC
"""

import re

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


with open('Retrieval/Retrieval-1.txt','r') as f:
    lines = f.readlines()

with open('Retrieval-1-a.txt', 'a+') as file:
    file.seek(0)
    for line in lines:
        splitQUERY = line.split('QUERY')
        print(splitQUERY)
        pngQuery = splitQUERY[0].split('png')
        if len(pngQuery) > 1:
            if pngQuery[1] == '\n':
                file.write(line)
            else:
                capts = split_on_uppercase(splitQUERY[0])

                string = 'query'+parts[0]+'png'+'\n'
                print(string)
                file.write(string)
                for i in capts:
                    string = i + '\n'
                    file.write(string)
                        
                print('\n')
                file.write('\n')
        elif splitQUERY[0] == '\n':
            file.write(line)
        else:
            file.write(line)
        
            
        
        
        
        
        '''
        for part in splitQUERY:
            if part != '':
                parts = part.split('query')
                
                for p in parts:
                    if len(p.split('r0')) > 1:
                        capts = p.split(' ', 1)
                        string = 'QUERY'+"".join([i for i in capts[0] if i.isdigit()]) + '\n'
                        print(string)
                        file.write(string)
                        print(capts)
                        capts1 = "".join([i for i in capts[0] if i.isalpha()]) + ' ' +capts[1]
                        cappart = capts1.split('r0')                     
                        capp = cappart[0]+'\n'
                        file.write(capp)
                        number = cappart[1].split('.')
                        number = number[0].split('-')
                        for i in range(0,10):
                            string = 'r'+str(i)+'-'+number[1]+'.png'+'\n'
                            print(string)
                            file.write(string)
                        
                        print('\n')
                        file.write('\n')                        
                        
                    elif len(p.split('r0')) == 1:
                        parts = p.split('png')
                        print(parts)
                        capts = split_on_uppercase(parts[1])

                        string = 'query'+parts[0]+'png'+'\n'
                        print(string)
                        file.write(string)
                        for i in capts:
                            string = i + '\n'
                            file.write(string)
                        
                        print('\n')
                        file.write('\n')
                    
'''