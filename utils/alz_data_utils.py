
MOCA_RANGE = [0,30]
MOCA_LEVELS = [0,20,26,30]
SA_CLASSES = ('SA-FALSE', 'SA-TRUE')
MOCA_CLASSES = ('DEMENTED', 'MILD', 'HEALTHY')

def moca_score_to_class(score):
    lbl = 0
    if MOCA_LEVELS[1] <= score:
        lbl = 1
    if MOCA_LEVELS[2] <= score:
        lbl = 2
    return lbl


def label_task_adjustment(score, task_type):
    label = None

    if task_type == 'MOCA_CLASSES_3':
        label = moca_score_to_class(score)
    elif task_type == 'MOCA_REGRESSION_15_PLUS':
        if score < 15:
            label = 15
    else: # moca regression ALL
        label = score

    return label
