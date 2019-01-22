from core.config import cfg

def get_vid_fid(im_name):
    im_name = im_name.split('/')[-1].split('.')[0]
    splits = im_name.split('_')
    fid = splits[-1]
    vid = ''
    for split in splits[:-1]:
        vid += (split+'_')
    vid = vid[:-1]
    return vid, fid


def check_sequence_break(prev_im_name, cur_im_name):
    prev_vid, prev_fid = get_vid_fid(prev_im_name)
    cur_vid, cur_fid = get_vid_fid(cur_im_name)
    if prev_vid != cur_vid:
        return True
    else:
        if int(cur_fid) < int(prev_fid):
            return True
        if (int(cur_fid) - int(prev_fid)) > cfg.CASCADE.ALLOWED_GAP:
            return True
    return False
 

def check_sequence_break_onlist(prev_list, cur_list):
    ret_list = []
    for i in range(len(prev_list)):
        for j in range(len(prev_list[i])):
            ret_list.append(check_sequence_break(prev_list[i][j], \
                                                    cur_list[i][j]))
    return ret_list


def split_blob_conv(blob_conv):
    # returns a dict
    out_dict = {}
    for l in range(len(blob_conv)):
        out_dict['blob_conv'+str(l)] = blob_conv[l]
    return out_dict


def lg_to_gl(blob_conv):
    levels = len(blob_conv)
    gpus = len(blob_conv[0])

    blob_conv_out = []
    for g in range(gpus):
        inner_blob_conv_out = []
        for l in range(levels):
            inner_blob_conv_out.append(blob_conv[l][g])
        blob_conv_out.append(inner_blob_conv_out)
    
    return blob_conv_out
