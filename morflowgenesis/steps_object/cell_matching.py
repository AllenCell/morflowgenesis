from morflowgenesis.utils.image_object import StepOutput
from prefect import task, get_run_logger, flow
import numpy as np
logger = get_run_logger()
import pandas as pd


def get_start_and_end_pts(roi):
    return np.asarray(roi)[::2], np.asarray(roi)[1::2]


def str_to_list(roi):
    return np.asarray(roi.split(',')).astype(int)

@task
def iou_from_roi(roi1, roi2, eps = 1e-8):
    roi1_starts, roi1_ends = get_start_and_end_pts(roi1)
    roi2_starts, roi2_ends = get_start_and_end_pts(roi2)

    max_start= np.max([roi1_starts, roi2_starts], axis = 0)
    min_end = np.min([roi1_ends, roi2_ends], axis = 0)
    intersection = np.prod(min_end - max_start)

    min_start = np.min([roi1_starts, roi2_starts], axis = 0)
    max_end = np.max([roi1_ends, roi2_ends], axis = 0)
    union = np.prod(max_end - min_start)
    return (intersection + eps) / (union + eps)

@flow
def match_cells(image_object, step_name, output_name, pred_step, label_step, iou_thresh = 0.9):
    if  image_object.step_is_run(f'{step_name}_{output_name}'):
        logger.info(f'Skipping step {step_name}_{output_name} for image {image_object.id}')
        return image_object
    pred_match_to_label= {'pred_cellid': [], 'label_cellid':[], 'iou':[]}
    pred_df = image_object.load_step(pred_step)
    label_df = image_object.load_step(label_step)
    for pred_row in pred_df.itertuples():
        best_iou = -np.inf
        best_cellid = None
        for label_row in label_df.itertuples():
            iou = iou_from_roi(str_to_list(pred_row.roi), str_to_list(label_row.roi))
            if iou > best_iou and iou>iou_thresh:
                best_iou = iou
                best_cellid = label_row.CellId
        if best_iou > iou_thresh:
            pred_match_to_label['pred_cellid'].append(pred_row.CellId)
            pred_match_to_label['label_cellid'].append(best_cellid)
            pred_match_to_label['iou'].append(best_iou)
        
    step_output = StepOutput(image_object.working_dir, step_name, output_name, output_type = 'csv', image_id = image_object.id)
    step_output.save(pd.DataFrame(pred_match_to_label)   )
    image_object.add_step_output(step_output)
    image_object.save()
    return image_object

