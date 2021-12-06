''' A dictionary of supported experiments '''

from experiments.experiment import Experiment
from experiments.peldor_4p_rect import Peldor_4p_rect
from experiments.ridme_5p_rect import Ridme_5p_rect
from experiments.peldor_4p_chirp import Peldor_4p_chirp

experiment_types = {}
experiment_types['4pELDOR-rect'] = Peldor_4p_rect
experiment_types['5pRIDME-rect'] = Ridme_5p_rect
experiment_types['4pELDOR-chirp'] = Peldor_4p_chirp