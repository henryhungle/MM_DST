import numpy as np 
from typing import List, Dict, Tuple
from functools import cmp_to_key 
import pdb 

def get_all_video_objects(object_labels: List[np.ndarray]) -> List[int]:
    video_objects = set()
    for frame_idx in range(len(object_labels)):
        frame_objects = object_labels[frame_idx]
        video_objects = video_objects.union(frame_objects)

    return list(video_objects)

def normalize_bb_label(label, vft, frame_shapes): 
    boxes = label['bb']
    objects = label['labels'] 

    norm_boxes = vft['norm_bb'] 
    object_mapping = vft['object_mapping']  
    num_pred_objects = vft['num_possible_objects']
    
    output = [] 
    label_masks = [] 
    num_frames = len(norm_boxes) 
    
    for frame_idx in range(num_frames):
        ft = norm_boxes[frame_idx]
        gt_boxes = np.zeros(ft.shape)
        mask = np.asarray([0] * len(gt_boxes), dtype=np.int32)
        if frame_idx == 0: # <frame> token position 
            output.append(gt_boxes)
            label_masks.append(mask)
            continue

        frame_objects = objects[frame_idx-1] 
        frame_boxes = boxes[frame_idx-1]
        object_and_bb_labels = {k:v for k,v in zip(frame_objects, frame_boxes)}
        for obj_idx in range(num_pred_objects): 
            obj_id = object_mapping[obj_idx] 
            if obj_id in object_and_bb_labels:
                bb_label = object_and_bb_labels[obj_id] 
                gt_boxes[obj_idx+1] = bb_label # position 0 is <frame> token 
                mask[obj_idx+1] = 1
        gt_boxes = gt_boxes / frame_shapes # normalize
        output.append(gt_boxes) 
        label_masks.append(mask) 
    
    vft['label_bb'] = output 
    vft['label_mask'] = label_masks 
    return vft 

def normalize_and_pad_bb(video_ft, max_objects, frame_shapes, vocab, temporal_grounding): 
    prediction_boxes = video_ft['bb']
    object_labels = video_ft['labels']
    
    padded_video_boxes: List[np.ndarray] = []
    padded_video_objs: List[np.ndarray] = []
    padded_video_encoded_objs: List[np.ndarray] = [] 
    padded_video_frames: List[np.ndarray] = [] 

    padding_bb = np.array([0] * 4)  

    padded_video_boxes.append(np.asarray([padding_bb])) 
    padded_video_objs.append(['<video>'])
    padded_video_encoded_objs.append([vocab['<video>']])
    padded_video_frames.append(np.asarray([0]))

    num_frames = len(object_labels)
    video_objects = get_all_video_objects(object_labels)
    sorted_objects = sorted(video_objects) 
    video_objects_order = {idx: label for idx, label in enumerate(sorted_objects)}
    num_possible_objects_video = min(len(video_objects_order), max_objects)
    
    for frame_idx in range(num_frames):

        padded_frame_boxes: List[np.ndarray] = [np.asarray(padding_bb)]
        if temporal_grounding:
            padded_frame_objs = ['<frame{}>'.format(frame_idx+1)] # start from frame 1 
        else:
            padded_frame_objs = ['<frame>'] 
        frame_objects: np.ndarray = object_labels[frame_idx]
        frame_predictions: np.ndarray = prediction_boxes[frame_idx]
        objects_and_bb: List[Tuple[int, np.ndarray]] = list(zip(frame_objects, frame_predictions))
        sorted_objects_and_bb = sorted(objects_and_bb, key=lambda x: x[0])
        num_objects_in_frame = len(sorted_objects_and_bb)

        current_object_idx = 0
        video_objects_order_idx = 0
        last_object = -1
        while current_object_idx < num_objects_in_frame:
            # if the video frame contains more than max objects we will discard the last ones
            # it is considered a limitation of the perception model
            if video_objects_order_idx >= num_possible_objects_video:
                break

            current_object, current_bb = sorted_objects_and_bb[current_object_idx]

            if current_object == video_objects_order[video_objects_order_idx]:
                # the objects are in the correct order (no missing objects)

                # add real bb bit set to 1, and also is cone bit
                object_tracks = np.asarray(current_bb)
                padded_frame_boxes.append(object_tracks)
                padded_frame_objs.append('<obj{}>'.format(current_object))
                current_object_idx += 1
                video_objects_order_idx += 1
                last_object = current_object

            elif current_object == last_object:
                # in case we have the same object id more than once this is a mistake of the perception model
                # we will ignore that object
                current_object_idx += 1

            else:
                # object is missing, needs to add padding instead (dummy coordinates) 
                # check if the object that is supposed to be in this index is a cone or not
                padded_frame_boxes.append(padding_bb)
                padded_frame_objs.append('<obj{}>'.format(video_objects_order[video_objects_order_idx]))

                video_objects_order_idx += 1
        
        # if necessary add more padding bb to match the num_possible_objects 
        num_current_frame_objects = len(padded_frame_boxes)-1 #minus the <frame> position 
        if num_current_frame_objects < num_possible_objects_video:
            required_padding_length = num_possible_objects_video - num_current_frame_objects
            padding_bbs = [padding_bb] * required_padding_length
            padding_objs = [] 
            for obj_idx in range(num_current_frame_objects, num_possible_objects_video):
                obj_id = video_objects_order[obj_idx] 
                padding_objs.append('<obj{}>'.format(obj_id))
            padded_frame_boxes.extend(padding_bbs)
            padded_frame_objs.extend(padding_objs)
    
        if len(padded_frame_boxes) != num_possible_objects_video+1: pdb.set_trace()
        if len(padded_frame_objs) != num_possible_objects_video+1: pdb.set_trace()
        # finally this is the bounding boxes samples for the current frame
        # pack as a numpy array with dimensions (10, 5)
        # normalize and then add to list videos frame bbs
        norm_frame_boxes = np.array(padded_frame_boxes)
        norm_frame_boxes = norm_frame_boxes / frame_shapes  # normalize
        padded_video_boxes.append(norm_frame_boxes)
        padded_video_objs.append(padded_frame_objs)
        encoded_objs = [vocab[t] for t in padded_frame_objs]
        padded_video_encoded_objs.append(encoded_objs) 
        turn_idx = np.asarray([frame_idx+1] * len(padded_frame_objs))
        padded_video_frames.append(turn_idx) 

    video_ft['norm_bb'] = padded_video_boxes
    video_ft['object_mapping'] = video_objects_order
    video_ft['objects'] = padded_video_objs 
    video_ft['encoded_objects'] = padded_video_encoded_objs 
    video_ft['frames'] = padded_video_frames
    video_ft['num_possible_objects'] = num_possible_objects_video 
    
    return video_ft  


def get_resnext_ft(video_ft, num_objs): 
    resnext_dim = 2048
    output = [np.array([[0]*resnext_dim])] # for <video> token position 
    segment_map = {0: (0,0)}
    for clip_idx, clip in enumerate(video_ft['clips']):
        # expanded by num_objs + 1 for the <frame> token position 
        expanded_ft = np.repeat(clip['features'][np.newaxis,:], num_objs+1, axis=0)
        output.append(expanded_ft)
        segment_map[clip_idx+1] = tuple(clip['segment']) 
    return output, segment_map































