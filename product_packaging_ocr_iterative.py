from datetime import datetime

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageOps

from quality_functions import iterate_detect_text
from quality_functions import annotate_image
from quality_functions import overlap
from quality_functions import visualize
 
   
def scan_images(image_list):

    all_annotations = {}
    
    for image_file in image_list:
        # Detect and grey out text on image, iterating until no new text is detected
        annotations = iterate_detect_text(image_file) 
        
        # Visualize greyed out detected text
        annotate_image(image_file, annotations, image_file.split('.')[0]+'_greyed_out.png', 1, (150,150,150), -1, 1)
    
        all_annotations[image_file] = annotations
        
    return all_annotations
          

def get_trame_words(trame_text_list):
    trame_words = []
    
    for trame in trame_text_list:
        with open(trame, encoding='latin-1') as file:  
            trame_text = file.read() 
            
        trame_text = trame_text.replace('\n',' ')
        trame_list = trame_text.split(' ')
        
        trame_words += trame_list
        
    return trame_words 

        
##########################################################
'''Set up variables'''
### Set up image files to examine
front_image = 'platter_front.png'
back_image =  'platter_back.png'

### Set trame-text files to use for comparison
trame_file = 'image_text_files/platter_image_text.txt'

##########################################################

t1 = datetime.now()

all_annotations = scan_images([front_image,back_image])

trame_words = get_trame_words([trame_file])

detected_words = [value for annotation_dict in all_annotations.values() for value in annotation_dict.values()]

results = overlap(trame_words,detected_words)

# Print results
print()
print('Percentage of trame text words detected:')
print(round((results[2]*100),2),'%')
print()

visualize(front_image,all_annotations[front_image],'blue')
visualize(back_image,all_annotations[back_image],'blue')

duration = datetime.now() - t1
print('Duration:',str(duration))