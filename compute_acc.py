import json 
import argparse  
from pathlib import Path
from datasets.dst_utils import * 
import glob 
from tqdm import tqdm 
import copy 

def iou_compute(gt, pred, frame_rate): 
    if gt == (-1,-1): # no segment output (e.g. spatial-only questions)
        if pred == (-1, -1):
            return 1 
        else:
            return 0 
    if pred == (-1, -1): # no predicted start frame and end frame  
        return 0
    gt_start, gt_end = gt[0], gt[1] 
    assert pred[0] != -1 
    pred_start = get_frame_idx(pred[0], frame_rate)
    if pred[1] == -1: 
        return 0 # no predicted end frame 
    else:
        pred_end = get_frame_idx(pred[1], frame_rate)
    intersection = range(max(gt_start, pred_start), min(gt_end, pred_end)+1)
    union = range(min(gt_start, pred_start), max(gt_end, pred_end)+1)
    iou = len(intersection)/len(union)
    return iou 
    
def iou_sort(iou, count):
    if iou <= 0.3: 
        return count 
    if iou > 0.3: 
        count[0.3] += 1 
    if iou > 0.5: 
        count[0.5] += 1 
    if iou > 0.7: 
        count[0.7] += 1 
    if iou > 0.9: 
        count[0.9] += 1
    if iou == 1.0: 
        count[1.0] += 1 
    return count 

def joint_match_sort(iou, gt, pred, count):
    if gt != pred:
        return count 
    if iou <= 0.3: 
        return count 
    if iou > 0.3: 
        count[0.3] += 1 
    if iou > 0.5: 
        count[0.5] += 1 
    if iou > 0.7: 
        count[0.7] += 1 
    if iou > 0.9: 
        count[0.9] += 1 
    if iou == 1.0: 
        count[1.0] += 1 
    return count 

def f1_compute(matches, num_gt, num_pred):
    recall = matches/num_gt if num_gt!=0 else 0
    precision = matches/num_pred if num_pred!=0 else 0 
    if recall+precision == 0: 
        f1 = 0 
    else:
        f1 =  2*recall*precision/(recall+precision)
    print("Recall: {} Precision: {} F1: {}".format(recall, precision, f1))
    return recall, precision, f1 

def find_num_slots(state, sizes, colors, materials, shapes):
    #sizes, colors, materials, shapes = 0, 0, 0, 0
    for k,v in state.items():
        for i,j in v.items():
            if i == 'SIZE': sizes += 1
            if i == 'COLOR': colors += 1
            if i == 'MATERIAL': materials += 1
            if i == 'SHAPE': shapes += 1
    return sizes, colors, materials, shapes

parser = argparse.ArgumentParser(description='compute performance metrics on DVD-DST for multimodal DST tasks')
parser.add_argument("--results", type=str, required=True, help='path to the output predicted states')
parser.add_argument('--frame_rate', type=int, required=False, default=12, help='sampling rate of video frames') 
parser.add_argument('--num_dials', type=int, required=False, default=1000000000000, help='total number of dialogues to be evaluated')
parser.add_argument('--by_turns', action='store_true')
args = parser.parse_args()

file_name = args.results 
results_raw = json.load(open(file_name, 'r'))
results = []
turn_indices = []
count = 0 
for k,v in results_raw.items():
    if count == args.num_dials: 
        break
    video_name = k 
    for i,j in v.items():
        turn = int(i) 
        results.append(j)
        turn_indices.append(turn)
    count += 1 

raw_matches = 0 
matches = 0

obj_matches = 0 
num_gt_objs = 0 
num_pred_objs = 0 

slot_matches = 0 
num_gt_slots = 0 
num_pred_slots = 0 
detection_matches = 0 

size_matches = 0
color_matches = 0
material_matches = 0 
shape_matches = 0 

num_gt_sizes, num_pred_sizes = 0, 0
num_gt_colors, num_pred_colors = 0, 0
num_gt_materials, num_pred_materials = 0, 0
num_gt_shapes, num_pred_shapes = 0, 0 

total_iou = 0 
iou_count = {0.3: 0, 0.5: 0, 0.7: 0, 0.9: 0, 1.0: 0} 

joint_matches = copy.deepcopy(iou_count)  

turn_detection_matches = {i:0 for i in range(10)}
turn_obj_matches = {i:0 for i in range(10)}
gt_turn_objs = {i:0 for i in range(10)}
pred_turn_objs = {i:0 for i in range(10)}
turn_slot_matches = {i:0 for i in range(10)}
gt_turn_slots = {i:0 for i in range(10)}
pred_turn_slots = {i:0 for i in range(10)}

turn_matches = {i:0 for i in range(10)}
turn_counts = {i:0 for i in range(10)}
turn_joint_matches = {i:copy.deepcopy(iou_count) for i in range(10)}


pred_key = 'pred_state'
for state_idx, state in enumerate(results): 
    turn_idx = turn_indices[state_idx] 
    turn_counts[turn_idx] += 1 
   
    if state['gt_state'] == state[pred_key]:
        raw_matches += 1 
    
    _, gt_state = process_state_seq(state['gt_state'])
    pred_segment, pred_state = process_state_seq(state[pred_key])
    gt_segment = state['gt_segment']

    if gt_state == pred_state: 
        matches += 1 
        turn_matches[turn_idx] += 1 

    for k,v in gt_state.items():
        if k in pred_state:
            if pred_state[k] == v: 
                obj_matches += 1 
                turn_obj_matches[turn_idx] += 1 
            for i,j in v.items(): 
                if i in pred_state[k] and pred_state[k][i]==j: 
                    slot_matches +=1 
                    turn_slot_matches[turn_idx] += 1 
                    if i == 'SIZE': size_matches += 1 
                    if i == 'COLOR': color_matches += 1 
                    if i == 'MATERIAL': material_matches += 1 
                    if i == 'SHAPE': shape_matches += 1 
            detection_matches += 1            
            turn_detection_matches[turn_idx] += 1
    
    iou = iou_compute(gt_segment, pred_segment, args.frame_rate)
    total_iou += iou
    iou_count = iou_sort(iou, iou_count) 

    num_gt_objs += len(gt_state) 
    num_pred_objs += len(pred_state)
    num_gt_slots += sum([len(v) for v in gt_state.values()])
    num_pred_slots += sum([len(v) for v in pred_state.values()])

    gt_turn_objs[turn_idx] += len(gt_state)
    pred_turn_objs[turn_idx] += len(pred_state)
    gt_turn_slots[turn_idx] += sum([len(v) for v in gt_state.values()])
    pred_turn_slots[turn_idx] += sum([len(v) for v in pred_state.values()])

    num_gt_sizes, num_gt_colors, num_gt_materials, num_gt_shapes =find_num_slots(
            gt_state, num_gt_sizes, num_gt_colors, num_gt_materials, num_gt_shapes)
    num_pred_sizes, num_pred_colors, num_pred_materials, num_pred_shapes = find_num_slots(
            pred_state, num_pred_sizes, num_pred_colors, num_pred_materials, num_pred_shapes) 

    joint_matches = joint_match_sort(iou, gt_state, pred_state, joint_matches) 

    turn_joint_matches[turn_idx] = joint_match_sort(iou, gt_state, pred_state, turn_joint_matches[turn_idx])
    
if True:
    raw_joint_acc = raw_matches/len(results)
    joint_acc = matches/len(results)
    print("Raw State Acc: {}".format(raw_joint_acc))
    print("Joint Obj State Acc: {}".format(joint_acc))
    joint_state_acc = {k: v/len(results) for k,v in joint_matches.items()}
    print("Joint State IoU@k: {}".format(joint_state_acc))
    print()
    print("Object Identity Prediction") 
    detection_recall, detection_precision, detection_f1 = f1_compute(detection_matches, num_gt_objs, num_pred_objs)
    print("Object Slot Prediction")
    slot_recall, slot_precision, slot_f1 = f1_compute(slot_matches, num_gt_slots, num_pred_slots)
    print("Object State Prediction")
    obj_recall, obj_precision, obj_f1 = f1_compute(obj_matches, num_gt_objs, num_pred_objs)
    print()
    print("Objective Size Prediction")
    size_recall, size_precision, size_f1 = f1_compute(size_matches, num_gt_sizes, num_pred_sizes)
    print("Objective Color Prediction")
    color_recall, color_precision, color_f1 = f1_compute(color_matches, num_gt_colors, num_pred_colors)
    print("Objective Material Prediction")
    material_recall, material_precision, material_f1 = f1_compute(material_matches, num_gt_materials, num_pred_materials)
    print("Objective Shape Prediction")
    shape_recall, shape_precision, shape_f1 = f1_compute(shape_matches, num_gt_shapes, num_pred_shapes)
    print() 

if args.by_turns:
    print("Results by Turn Position") 
    for k,v in turn_slot_matches.items():

        print("Turn: {}".format(k+1))
        print("Object Identity Prediction") 
        turn_detection_recall, turn_detection_precision, turn_detection_f1 = f1_compute(
            turn_detection_matches[k], gt_turn_objs[k], pred_turn_objs[k])
        print("Object SLot Prediction")
        turn_recall, turn_precision, turn_f1 = f1_compute(
            v, gt_turn_slots[k], pred_turn_slots[k])
        print("Object State Prediction")
        turn_obj_recall, turn_obj_precision, turn_obj_f1 = f1_compute(
            turn_obj_matches[k], gt_turn_objs[k], pred_turn_objs[k])
        
        joint_acc = turn_matches[k]/turn_counts[k]
        print("Joint Obj State Acc: {}".format(joint_acc))
        turn_joint_acc = {i:j/turn_counts[k] for i,j in turn_joint_matches[k].items()}
        print("Joint State IoU@k: {}".format(turn_joint_acc))


    



