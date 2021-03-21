def save_score(score, filepath):
    ''' Saves the score as a function of optimization step '''
    file = open(filepath, 'w')
    for i in range(score.size):
        file.write('{0:<15d} {1:<15.6f} \n'.format(i+1, score[i]))
    file.close()