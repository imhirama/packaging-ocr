# -*- coding: utf-8 -*-
import io
import os
import numpy as np
import cv2 as cv
import csv
import string

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageOps

import Levenshtein as lv
from matplotlib.font_manager import findfont


# Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


# Detect text in a given image, return a dictionary of text and coordinates 
def detect_text(image):
    path = os.path.join(image)
    
    outtext  = open('output_text.txt', 'w') 
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print()
    
    print('Text detected:')
    
    box_dict = {}
    
    count = 0

    for text in texts:
        count += 1
        # Prevent saving first big block:       
        if count > 1:         

            print('{}'.format(text.description), end=', ')
    
            vertices = ([(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])
    
            #print('bounds: {}'.format(','.join(vertices)))
            #print(text.description)
            
            box_dict[tuple(vertices)] = text.description
        
            outtext.write(str(text))
    
    print()    
    outtext.close()
    
    #print(boxes)
    
    return box_dict
    
    
# Detect text in a given image as a single block 
def detect_text_as_block(image):
    path = os.path.join(image)
    
    outtext  = open('output_text.txt', 'w') 
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print()
    
    print('Text detected:')
    
    box_dict = {}
    
    
    count = 0

    for text in texts:   
        while count == 0:

            print('{}'.format(text.description), end=', ')
    
            vertices = ([(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])
       
            box_dict[tuple(vertices)] = text.description
        
            outtext.write(str(text))
            
            count += 1
    
    print()    
    outtext.close()
        
    return box_dict
    
    
# Open an image and draw boxes on it from a dictionary,
# choosing zoom, box color, fill, and padding
def annotate_image(image_file, dictionary, file_name, zoom, color, border_or_fill, padding, show = False, wait = 1):

    img = cv.imread(image_file)
    
    # Draws rectangles
    for box in dictionary.keys():
        img = cv.rectangle(img, tuple(map(sum, zip(box[0],(
        -padding,-padding)))), tuple(map(sum, zip(box[2],(
        padding, padding)))), color, border_or_fill)
                

    # Saves output image   
    cv.imwrite(file_name,img)
    
    if show:
        cv.namedWindow('output', cv.WINDOW_NORMAL)
        cv.resizeWindow('output', int(img.shape[1]*zoom),int(img.shape[0]*zoom))
        cv.imshow('output',img)
        cv.waitKey(wait)
        
    return file_name
    
    
# Add text to image using PIL
def add_text(background_image, annotation_dict, output_file,text_color = 'black'):
    image = Image.open(background_image)
    draw = ImageDraw.Draw(image)
    
    fontsize = [60,30,12,10,9]
    
    for box in annotation_dict.keys():
            txt = annotation_dict[box]
            width = abs(box[0][0]-box[1][0])
            height = abs(box[0][1]-box[2][1])
            font_index= 0
            font_file = findfont("arial")

            font = ImageFont.truetype(font_file,fontsize[font_index]) # starting font
            
            # Heuristic to check if text is horizontal or vertical
            if height < 3*width or len(txt) == 1:
                
                # Regular horizontal text
                while font.getsize(txt)[1] > height or font.getsize(txt)[0] > width:
                    # Try smaller fonts until text fits in box
                    try: 
                        font = ImageFont.truetype(font_file,fontsize[font_index])
                        font_index += 1
                    except:
                        font = ImageFont.truetype(font_file,8)
                        break
                        
                draw.text(box[0],txt,(text_color),font = font)
                   
            else:
                #print('Vertical', txt)
                # Vertical text 
                vertical_text(box, txt, image)

    image.save(output_file)
    
    return output_file
    
    
# Create a text box and rotates it 90 degrees for vertical text, adds to image
def vertical_text(box, text, PIL_image):
    font_file = findfont("arial")

    font = ImageFont.truetype(font_file,15)
    
    #width = abs(box[0][0]-box[1][0])
    #height = abs(box[0][1]-box[2][1])
    
    vert_box=Image.new('L', (100,100))
    
    draw_box = ImageDraw.Draw(vert_box)
    
    draw_box.text( (0, 0), text,  font=font, fill=255)
    
    vert_box=vert_box.rotate(90,  expand=1)
    
    PIL_image.paste( ImageOps.colorize(vert_box, (0,0,0), (0,0,0)), box[0],  vert_box)
    
    #PIL_image.show(PIL_image)
    
    return text
    

########################### Panels


# Takes an image and its annotation dictionary and detects either vertical
# or horizontal text panels with a specified gap size between them (in pixels)
def detect_panels(image_name, annotation_dict, gap_size, orientation = 'vertical', show = False):
        
    if type(image_name) == str:
        image =Image.open(image_name)
    else:
        image = image_name
    
    draw = ImageDraw.Draw(image)
    
    (im_w,im_h) = image.size
    
    tile_dim = gap_size
            
    if orientation == 'horizontal':
        x_or_y = 1
    else:
        x_or_y = 0
   
    
    x_list =[]
    for point in annotation_dict.keys():
        start = min(point[0][x_or_y], point[2][x_or_y])
        stop = max(point[0][x_or_y],point[2][x_or_y])
        while start < stop:
            x_list.append(start)
            start += round(tile_dim/2)
        x_list.append(stop)
        
    x_list.sort()
      
    x_lines = []
    
    for x in range(1,len(x_list)):
        if abs(x_list[x] - x_list[x-1]) > tile_dim: 
            x_lines.append(round((x_list[x]+x_list[x-1])/2))
            
    if show:   
        
        if orientation == 'horizontal':
            for coord in x_lines:
                draw.line([(0,coord),(im_w,coord)], fill=(255,80,0), width=7)
        else:
            for coord in x_lines:
                draw.line([(coord,0),(coord,im_h)], fill=(255,80,0), width=7)
                
        image.show(image)
    
    return x_lines
    

# Splits an image into multiple panels, vertically or horizontally,
# based on a list of split x or y coordinates
def split_image(image_file_name, split_list, orientation = 'vertical'):
    
    if type(image_file_name) == str:
        image = Image.open(image_file_name)
    else:
        image = image_file_name
        
    (im_w,im_h) = image.size
    
    start = 0
    
    if orientation == 'horizontal':
        end = im_h
    else:
        end = im_w
    
    # Add beginning and end of image to split list
    split_list.insert(0, start)
    split_list.append(end)
    
    panels = {}
    
    for k in range(1,len(split_list)):
                
        if orientation == 'horizontal':
            panel = image.crop((0, split_list[k-1], im_w, split_list[k]))
        else:
            panel = image.crop((split_list[k-1], 0, split_list[k], im_h))
        
        #Add new panels to dictionary where key is a tuple of starting and ending
        #coordinates, value is the panel image
        panels[(split_list[k-1],split_list[k])] = panel
                
    return panels
    

def iterate_detect_text(image):
    print('Examining image',image)
    
    # Create an empty dictionary for collecting all annotations
    annotations  = {}
    
    # Create a list to keep track of the number of annotations added per round
    additions = []
    
    # Instantiate a client
    #client = vision.ImageAnnotatorClient()
    #print('Client instatiated')
    
    # Prepare variables for looping
    latest_image = image
    iterations = 0
    boxes = {} 
    
    # Call detect_text and annotate_image functions to detect text and grey it out,
    # repeating until no more text is recognized:
    print('Starting Text Detection')
    while len(boxes) > 0 or iterations == 0:
        boxes = detect_text(latest_image)
        latest_image = annotate_image(latest_image, boxes, 'grey.png', 1, (150,150,150), -1, 1, wait = 20)
        
        # Add latest detect text boxes to master dictionary
        annotations.update(boxes)
        additions.append(len(boxes))
        
        iterations += 1
        
    # Print final text results of detection  
    print()
    print('-------------------------')
    print('No more text identified on iteration #%s' %iterations)
    print()
    print('Number of text instances recognized per iteration:')
    print(additions)
    
    return annotations
        
# Takes the start and stop coordinates of panel (in one dimension)
# and filters an annotation dictionary for only annotations within those panel coordinates 
def words_in_panel(panel_coordinates,annoations_dict,orientation='vertical'):
    panel_start,panel_end = panel_coordinates
    
    if orientation == 'vertical':
        panel_words  = {
            location: annoations_dict[location]
            for location in annoations_dict.keys()
            if location[0][0] >= panel_start
            and location[1][0] <= panel_end
            }
    else:
        panel_words  = {
            location: annoations_dict[location]
            for location in annoations_dict.keys()
            if location[1][1] >= panel_start
            and location[0][1] <= panel_end
            }
        
    return panel_words

def calculate_overlap(trame_list, detected_list):
    print('--------------------------------------------------------')
    
    
    print('Number of words in trame text')
    print(len(set(trame_list)))
    
    print('Number of words detected')
    print(len(set(detected_list)))
    
    print('--------------------------------------------------------')
    
    print('Number of words in trame text and detected:')
    print(len(set(trame_list).intersection(set(detected_list))))
    
    print('Number of words in trame text but NOT detected:')
    print(len(set(trame_list)-(set(detected_list))))
    
    
    print('Number of words detected but NOT in trame text:')
    print(len(set(detected_list)-(set(trame_list))))
    
    
    print('Number of words in only one of the lists:')
    print(len(set(trame_list).symmetric_difference(set(detected_list))))
    
    print('--------------------------------------------------------')
    
    print('Proportion of trame text words detected:')
    print(len(set(trame_list).intersection(set(detected_list)))/len(trame_list))
    print()
    print('Proportion of detected words in trame text:')
    print(len(set(trame_list).intersection(set(detected_list)))/len(detected_list))
    
    
    print('--------------------------------------------------------')
    
    not_detected = set(trame_list).difference(set(detected_list))
    detected = set(trame_list).intersection(set(detected_list))
    percent_detected = len(set(trame_list).intersection(set(detected_list)))/len(set(trame_list))
    
    return (not_detected,detected,percent_detected)
      

def overlap(ref_list, detected_list):
    print('--------------------------------------------------------')
    
    # Remove punctuation at beginning and end of words, and remove punctuation-only strings
    ref_list = [word.strip(string.punctuation) for word in ref_list]
    ref_list = list(filter(None, ref_list))
    
    detected_list = [word.strip(string.punctuation) for word in detected_list]
    detected_list = list(filter(None, detected_list))
    
    print('Number of words in reference text')
    print(len(ref_list))
    
    print('Number of words detected')
    print(len(detected_list))
    
    print('--------------------------------------------------------')
    
    ref_and_detected = []
    ref_not_detected = []
    
    detected_remaining = [word for word in detected_list]
    
    for word in ref_list:
        if word in detected_remaining:
            ref_and_detected.append(word)
            detected_remaining.remove(word)
        else:
            ref_not_detected.append(word)
    
    detected_not_ref = detected_remaining    
    
    print('Number of words in reference text and detected list:')
    print(len(ref_and_detected))
    
    print('Number of words in reference text but NOT detected:')
    print(len(ref_not_detected))
    
    
    print('Number of words detected but NOT in reference text:')
    print(len(detected_not_ref))
    
    
    print('Number of words in only one of the lists:')
    print(len(ref_not_detected)+len(detected_not_ref))
    
    print('--------------------------------------------------------')
    
    print('Proportion of reference text words detected:')
    percent_detected = len(ref_and_detected)/len(ref_list)
    print(percent_detected)
    print()
    print('Proportion of detected words in reference text:')
    print(len(ref_and_detected)/len(detected_list))
    
    
    print('--------------------------------------------------------')
    

    return (ref_not_detected,ref_and_detected,percent_detected)
 
def visualize(image_file, annotations, text_color):
      
    # Open the image and draw white boxes as background,
    # with slight padding so they fully cover existing text
    annotate_image(image_file, annotations, 'no_text.png', 1, (255,255,255), -1, 5)
    
    # Create blank background to write text on
    img = cv.imread(image_file)
    blank = np.zeros((img.shape[0], img.shape[1],3), np.uint8)
    blank[:,0:img.shape[1]] = (255,255,255)
    cv.imwrite('background.png',blank)
        
    # Call add_text to draw text on the image and on blank background  
    viz1 = add_text('no_text.png', annotations, image_file.split('.')[0]+'_viz1.png',text_color)
    viz2 = add_text('background.png', annotations, image_file.split('.')[0]+'_viz2.png',text_color)  
    
    return(viz1,viz2)
    