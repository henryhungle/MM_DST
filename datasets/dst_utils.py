import pdb 
import copy 
import re 
from object_attributes import *  

TRACKED_OBJECT_ATTRIBUTES = ['<Z>', '<C>', '<M>', '<S>', 'original_turn'] 
OTHER_ATTRIBUTES=[
        '<Z2>', '<C2>', '<M2>', '<S2>',
        '<Z3>', '<C3>', '<M3>', '<S3>',
        '<Z4>', '<C4>', '<M4>', '<S4>'
        ]
VIDEO_NUM_FRAMES = 300 
QUERY_TO_ATTRIBUTES = {
        "query_color": '<C>',
        "query_shape": '<S>',
        "query_size": '<Z>',
        "query_material": '<M>',
        }

def get_state_diff(prior_state, new_state): 
        state_diff = {} 
        for k,v in new_state.items():
            if k not in prior_state:
                state_diff[k] = v 
            else:
                prior_set = set(prior_state[k].items())
                new_set = set(new_state[k].items())
                diff_set = prior_set ^ new_set
                if len(diff_set) > 0: 
                    state_diff[k] = dict(diff_set) 
        return state_diff 

def get_start_time(event):
    if event is None: return 0
    return event[-1]

def get_end_time(event): 
    if event is None: return VIDEO_NUM_FRAMES
    return event[-1] 

def get_temporal_boundary(video_period): 
    if video_period is None: return None
    return get_start_time(video_period[0]), get_end_time(video_period[1])

def merge_temporal(cumulative_temporal, current_temporal): 
    pdb.set_trace()

def clean_state(state):
    new_state = {}
    for k,v in state.items():
        for v1,v2 in v.items():
            if v1 not in TRACKED_OBJECT_ATTRIBUTES: 
                if v1 in OTHER_ATTRIBUTES:
                    v1 = v1[:2] + v1[3:]
                else:
                    continue 
            if k not in new_state:
                new_state[k] = {}
            if v1 in new_state[k]: 
                if new_state[k][v1]!= v2: pdb.set_trace()
            new_state[k][v1] = v2 
    return new_state 

def get_question_type(template, prior_template):
    last_node_type = template['nodes'][-1]['type']
    text = template['text'][0].lower()
    if 'same set of activities' in text:
        if 'how many' in text:
            qtype = 'compare action set (count)'
        else:
            qtype = 'compare action set (exist)'
    elif 'same sequence of activities' in text:
        if 'how many' in text:
            qtype = 'compare action sequence (count)'
        else:
            qtype = 'compare action sequence (exist)'
    elif 'frequently' in text:
        qtype = 'compare int'
    elif 'how many times' in text:
        qtype = 'action count'
    elif 'how many' in text or 'what number' in text:
        qtype = 'obj count'
    elif 'is there' in text:
        qtype = 'obj exist'
    elif 'what color' in text or 'what material' in text or 'what shape' in text or 'what size' in text:
        qtype = 'attr query'
    elif 'what type of action' in text or 'what is the' in text or 'what types of action' in text:
        qtype = 'action query'
    else:
        assert 'what about' in text
        qtype = get_question_type(prior_template, None)
    return qtype

# remove state attributes that incorporate current answers
# only for questions of attribute query, object count, object exists
def remove_answer_obj_from_state(prior_state, state, template, prior_template, program):
    qtype = get_question_type(template, prior_template)
    new_state = copy.deepcopy(state)
    if qtype == 'attr query':
        assert program[-2]['type'] in ['unique', 'refer_object']
        oid = program[-2]['_output']
        attribute = QUERY_TO_ATTRIBUTES[program[-1]['type']]
        value = program[-1]['_output']
        assert new_state[str(oid)][attribute] == value
        if str(oid) in prior_state:
            new_state[str(oid)] = prior_state[str(oid)]
        else:
            del new_state[str(oid)]
    elif qtype in ['obj count', 'compare_action_set (count)', 'compare action sequence (count)'] and program[-1]['_output'] == 1:
        oid = program[-2]['_output']
        assert len(oid)==1
        oid = oid[0]
        new_state = remove_answer_obj(oid, prior_state, state, new_state, template)
    elif qtype in ['obj exist', 'compare action set (exist)', 'compare action sequence (exist)'] and program[-1]['_output']== True:
        if len(program[-2]['_output'])==1:
            oid = program[-2]['_output'][0]
            new_state = remove_answer_obj(oid, prior_state, state, new_state, template)
    return new_state

def remove_answer_obj(oid, prior_state, state, new_state, template):
    if 'temporal_obj_id' in template:
        temporal_oid = template['temporal_obj_id']
    else:
        temporal_oid = -1
    if temporal_oid != oid:
        if str(oid) in prior_state:
            new_state[str(oid)] = prior_state[str(oid)]
        else:
            if str(oid) in new_state:
                del new_state[str(oid)]
            else:
                pdb.set_trace()
    else:
        temporal_obj_attr = template['temporal_obj_attr']
        for k,v in state[str(oid)].items():
            if k!='original_turn' and (k not in temporal_obj_attr) and \
                    (str(oid) not in prior_state or k not in prior_state[str(oid)]):
                del new_state[str(oid)][k]
    return new_state

def remove_future_obj_from_state(prior_state, state, next_turn, next_turn_idx):
    new_state = copy.deepcopy(state)
    if 'among' in next_turn['question'] and next_turn['turn_dependencies']['object']=='last_unique':
        if 'sampled_ans_object_attr' not in next_turn['template']:
            pdb.set_trace()
        values = next_turn['template']['sampled_ans_object_attr']
        oid = next_turn['template']['sampled_ans_object']
        for k,v in state[str(oid)].items():
            if k in values:
                assert values[k] == v
                del new_state[str(oid)][k]
        if new_state[str(oid)]['original_turn'] == next_turn_idx+1:
            del new_state[str(oid)]
    return new_state


def get_slot(token):
    for k,v in OBJECT_ATTRIBUTES.items():
        if token in v: 
            slot = k[:-1].upper() 
            return slot 

def find_digit(string):
    return int(re.search(r'\d+', string).group())

def process_state_seq(state_seq):
    tokens = state_seq.split()
    curr_obj = None 
    curr_slot = None 
    obj_pattern = re.compile('<obj\d+>')
    frame_pattern = re.compile('<frame\d+>')
    state = {}
    start_frame = -1
    end_frame = -1 
    for token in tokens: 
        if frame_pattern.match(token): 
            if start_frame == -1: 
                start_frame = find_digit(token)
            elif end_frame == -1: 
                end_frame = find_digit(token) 
            else:
                pass # ignore this token 
        elif obj_pattern.match(token):
            curr_obj = token 
            if curr_obj not in state:
                state[curr_obj] = {} 
                curr_slot = None #refresh slot 
        elif token in ['SIZE', 'COLOR', 'MATERIAL', 'SHAPE']: 
            if curr_obj is None: 
                pass # no assigned object, skipping this token 
            else:
                curr_slot = token
        elif token == '<end_state>':
            break 
        else:
            slot_type = get_slot(token)
            if curr_obj is not None and curr_slot is not None: 
                # safety check 
                if curr_slot == slot_type: # safety check 
                    state[curr_obj][curr_slot] = token
                else:
                    state[curr_obj][slot_type] = token 
                curr_slot = None # refresh slot 
            else:
                pass #no assigned object/slot, skipping this token 
    return (start_frame, end_frame), state

def get_decoded_state(state):
    output = {} 
    for k,v in state.items(): 
        obj_id = str(find_digit(k))
        output[obj_id] = {} 
        for i,j in v.items(): 
            if i == 'SIZE': 
                output[obj_id]['<Z>'] = j
            elif i == 'COLOR': 
                output[obj_id]['<C>'] = j
            elif i == 'MATERIAL': 
                output[obj_id]['<M>'] = j
            elif i == 'SHAPE': 
                output[obj_id]['<S>'] = j 
    return output 

def get_frame_idx(frame, frame_rate):
    return (frame-1)*frame_rate + 1 

