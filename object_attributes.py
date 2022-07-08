from object_indices import *
import json 

OBJECT_SIZES = set()
OBJECT_COLORS = set()
OBJECT_MATERIALS = set()
OBJECT_SHAPES = set()

for k in OBJECTS_NAME_TO_IDX.keys(): 
    name = k.split('_')
    OBJECT_SIZES.add(name[0]) 
    OBJECT_COLORS.add(name[1])
    OBJECT_MATERIALS.add(name[3])
    OBJECT_SHAPES.add(name[2])

OBJECT_ATTRIBUTES = {
    "sizes": sorted(list(OBJECT_SIZES)),
    "colors": sorted(list(OBJECT_COLORS)),
    "materials": sorted(list(OBJECT_MATERIALS)),
    "shapes": sorted(list(OBJECT_SHAPES))
}

for k,v in OBJECT_ATTRIBUTES.items(): 
    OBJECT_ATTRIBUTES[k] = ['dontcare'] + ['none'] + v
    
#json.dump(OBJECT_ATTRIBUTES, open('object_attributes.json', 'w'), indent=4)
