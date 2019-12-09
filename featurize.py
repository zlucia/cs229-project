import numpy as np
import drawSvg as draw

"""
Function: parse_svg_path

Usage: parse_svg_path(YOUR_SVG_GLYPH_STRING)
------------------------------------
`path_string`: path (string) for a raw SVG glyph instruction
`draw_commands` = possible commands being considered for parsing SVG

Returns `as_list`, the instructions in list and float form (indexed instructions rather than characters).
"""
def parse_svg_path(path_string, draw_commands="MmZzLlHhVvCcSsQqTtAa"):
    
    def parse_helper(path_data):
        # Adapted from codereview.stackexchange.com/questions/28502/svg-path-parsing/
        digits = '0123456789eE'; whitespace = ', \t\n\r\f\v'; sign = '+-'; exponent = 'eE'
        flt = False; entity = ''
        for char in path_data:
            if char in digits:
                entity += char
            elif char in whitespace and entity:
                yield entity
                flt = False; entity = ''
            elif char in draw_commands:
                if entity:
                    yield entity
                    flt = False; entity = ''
                yield "~" + char
            elif char == '.':
                if flt:
                    yield entity
                    entity = '.'
                else:
                    flt = True; entity += '.'
            elif char in sign:
                if entity and entity[-1] not in exponent:
                    yield entity
                    flt = False; entity = char
                else:
                    entity += char
        if entity:
            yield entity
    
    codes = {draw_commands[i]: i for i in range(len(draw_commands))}
    def to_float(x):
        try:
            return float(x)
        except:
            return float(codes[x])
    
    as_str = ','.join([i for i in parse_helper(path_string)]).split("~")[1:]
    as_list = [[to_float(x) for x in s.split(",")[:-1]] if len(s) > 1 else [to_float(s)] for s in as_str]

    return as_list

"""
Function: get_feature_vector

Usage: get_feature_vector(parse_svg_path(YOUR_SVG_GLYPH_STRING))
------------------------------------
Generates the feature vector `B` in Chapters 3, 6 inspired by work from
https://dspace.mit.edu/bitstream/handle/1721.1/119692/1078149677-MIT.pdf?sequence=1&isAllowed=y

`svg_as_list`: list of lists of SVG commands and corresponding arguments in float form for a single glyph
`feat_vec_size`: length of feature vector to return, currently defined for length 9 feature vector

Ordered list of commands upon which the indexing is based: `MmZzLlHhVvCcSsQqTtAa`

Returns `feat_vecs`, a numpy array of shape (num_instructions, feat_vec_size) containing normalized
feature vectors for the given glyph, all standardized to length 9 as below using the cubic SVG convention.

Feature vectors are generated as follows:

     Denote `s` = start point, `e` = end point, `c1`/`c2` = control points 1 and 2 respectively,
     
     9-dimensional feature vector for a single command = [disp(s, c1), disp(c1, c2), disp(c2, e), pen_state]
     
     `disp(a, b)` represents the displacement between control points a and b (a precedes b in the instructions),
     a numpy array of 2 elements representing displacement of x, y directions respectively.

     The first 6 elements of the feature vector are these three displacements, standardized across all commands
     to be in the format of the required arguments to the cubic SVG command (zeros where not applicable).
     
     `pen_state` has three binary elements: pen_up, pen_down, end drawing (in that order).

Helpful resource on instruction definitions: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d
"""
def get_feature_vector(svg_as_list, feat_vec_size=9):
    curr_pos = np.zeros(2)  # current coordinate position (x, y)
    init_pos = np.array(svg_as_list[0][1:])  # initial SVG starting point

    feat_vecs = []  # the sequence of instructions for a single glyph; contains one numpy feature vector per instruction

    prev_ctrl_pt = None  # previous control point, if any
    prev_ctrl_q = False  # whether previous control point was quadratic

    for seg in svg_as_list:
        seg = np.array(seg)
        feat = np.zeros(feat_vec_size)  # feature vector for current instruction

        if seg[0] == 12 or seg[0] == 13 or seg[0] == 18 or seg[0] == 19:  # S, s, A, a
            continue
        elif seg[0] == 0:  # M
            feat[:2] = seg[1:] - curr_pos  # displacement from current location
            curr_pos = seg[1:]

            feat[6:] = np.array([1., 0., 0.])  # pen up
        elif seg[0] == 1:  # m
            feat[:2] = seg[1:]
            curr_pos += seg[1:]        

            feat[6:] = np.array([1., 0., 0.])  # pen up
        elif seg[0] == 2 or seg[0] == 3:  # Z
            feat[:2] = init_pos - curr_pos
            curr_pos = init_pos

            feat[6:] = np.array([0., 1., 1.])  # pen down and end drawing
        elif seg[0] == 4:  # L
            feat[:2] = seg[1:] - curr_pos  # displacement from current location
            curr_pos = seg[1:]

        elif seg[0] == 5:  # l
            feat[:2] = seg[1:]
            curr_pos += seg[1:]

        elif seg[0] == 6:  # H
            feat[:2] = np.array([seg[1] - curr_pos[0], 0.])  # displacement on x only
            curr_pos[0] = seg[1]

        elif seg[0] == 7:  # h
            feat[:2] = np.array([seg[1], 0.])  # displacement on x only
            curr_pos[0] += seg[1]

        elif seg[0] == 8:  # V
            feat[:2] = np.array([0., seg[1] - curr_pos[1]])  # displacement on y only
            curr_pos[1] = seg[1]

        elif seg[0] == 9:  # v
            feat[:2] = np.array([0., seg[1]])  # displacement on y only
            curr_pos[0] += seg[1]

        elif seg[0] == 10:  # C
            feat[:2] = seg[1:3] - curr_pos  # displacement to 1st control point
            feat[2:4] = seg[3:5] - seg[1:3]  # displacement to 2nd control point
            feat[4:6] = seg[5:7] - seg[3:5]  # displacement to final control point
            curr_pos = seg[5:7]

        elif seg[0] == 11:  # c
            feat[:2] = seg[1:3]  # displacement to 1st control point
            feat[2:4] = seg[3:5] # displacement to 2nd control point
            feat[4:6] = seg[5:7] # displacement to final control point
            curr_pos += seg[5:7]

        elif seg[0] == 14:  # Q
            feat[:2] = seg[1:3] - curr_pos  # displacement to 1st control point
            feat[2:4] = seg[3:5] - seg[1:3]  # displacement to final control point
            curr_pos = seg[3:5]

            prev_ctrl_q = True
            prev_ctrl_pt = seg[1:3]
        elif seg[0] == 15:  # q
            feat[:2] = seg[1:3]  # displacement to 1st control point
            feat[2:4] = seg[3:5]  # displacement to final control point
            curr_pos += seg[3:5]

            prev_ctrl_q = True
            prev_ctrl_pt = seg[1:3]
        elif seg[0] == 16:  # T
            if not prev_ctrl_q:  # previous command wasn't quadratic
                new_ctrl_pt = curr_pos
                feat[:2] = np.zeros(2)  # displacement is 0, first control point is curr_pos
                feat[2:4] = seg[1:3] - curr_pos  # displacement of final control point from curr_pos            
            else:  # previous command was quadratic
                new_ctrl_pt = (curr_pos - prev_ctrl_pt) + curr_pos
                feat[:2] = new_ctrl_pt - curr_pos
                feat[2:4] = seg[1:3] - new_ctrl_pt
            curr_pos = seg[1:3]

            prev_ctrl_q = True
            prev_ctrl_pt = new_ctrl_pt
        elif seg[0] == 17:  # t
            if not prev_ctrl_q:  # previous command wasn't quadratic
                new_ctrl_pt = curr_pos
                feat[:2] = np.zeros(2)  # displacement is 0, first control point is curr_pos
                feat[2:4] = seg[1:3]  # displacement of final control point from curr_pos
            else:  # previous command was quadratic
                new_ctrl_pt = (curr_pos - prev_ctrl_pt) + curr_pos
                feat[:2] = new_ctrl_pt - curr_pos
                feat[2:4] = seg[1:3] - new_ctrl_pt
            curr_pos += seg[1:3]

            prev_ctrl_q = True
            prev_ctrl_pt = new_ctrl_pt

        if seg[0] >= 4:  # all pen down actions
            feat[6:] = np.array([0., 1., 0.])  # pen down

        if seg[0] not in [14, 15, 16, 17]:  # previous point wasn't quadratic
            prev_ctrl_pt = None
            prev_ctrl_q = False

        feat_vecs.append(feat)
    feat_vecs = np.stack(feat_vecs, axis=0)
    feat_vecs[:, :6] /= 5000  # normalize to reasonable range for RNN
    feat_vecs = feat_vecs[:, [0,1,2,3,6,7,8]]
    return feat_vecs


if __name__ == "__main__":
    # example of usage
    svg_example = "M169 0v1456h476q237 0 356.5 -98t119.5 -290q0 -102 -58 -180.5t-158 -121.5q118 -33 186.5 -125.5t68.5 -220.5q0 -196 -127 -308t-359 -112h-505zM361 681v-524h317q134 0 211.5 69.5t77.5 191.5q0 263 -286 263h-320zM361 835h290q126 0 201.5 63t75.5 171q0 120 -70 174.5t-213 54.5h-284v-463z"
    roboto_regular_B = get_feature_vector(parse_svg_path(svg_example))
    print(roboto_regular_B)
    print('list of feature vectors shape:', roboto_regular_B.shape)
