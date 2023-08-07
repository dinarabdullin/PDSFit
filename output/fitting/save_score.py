import numpy as np

def save_score(score, filepath):
    ''' Saves the score as a function of optimization step '''
    file = open(filepath, 'w')
    for i in range(score.size):
        file.write('{0:<20d} {1:<20.1f} \n'.format(i+1, score[i]))
    file.close()

    
def save_score_all_runs(score_all_runs, filepath):
    ''' Saves the score as a function of optimization step for multiple runs '''
    file = open(filepath, 'w')
    yt = []
    ymin = []
    count = 1
    for r in range(len(score_all_runs)):
        y = score_all_runs[r]
        count += y.size
        ymin.append(np.amin(y))
        yt.append(y)
    xt = np.arange(1, count+1, 1)
    yt = [item for sublist in yt for item in sublist]
    yt.append(np.amin(ymin))
    yt = np.array(yt)
    for i in range(xt.size):
        file.write('{0:<20d} {1:<20.1f} \n'.format(xt[i], yt[i]))
    file.close()