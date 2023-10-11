import numpy as np


def save_score(filepath, score):
    """Save goodness-of-fit vs. optimization step."""
    file = open(filepath, "w")
    for i in range(score.size):
        file.write("{0:<20d}{1:<20.1f}\n".format(i + 1, score[i]))
    file.close()

    
def save_score_all_runs(filepath, score_all_runs):
    """Save goodness-of-fit vs. optimization step for several optimizations."""
    joint_score = np.concatenate(score_all_runs, axis = None)
    save_score(filepath, joint_score)