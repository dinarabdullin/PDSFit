class Simulator():
    '''Simulator class '''

    def __init__(self, calculation_settings):
        self.distributions = calculation_settings['distributions']
        self.excitation_threshold = calculation_settings['excitation_treshold']
        self.euler_angles_convention = calculation_settings['euler_angles_convention']
        self.background = calculation_settings['background']