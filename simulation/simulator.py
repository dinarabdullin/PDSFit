class Simulator():
    '''Simulator class '''

    def __init__(self, calculation_settings):
        self.distributions = calculation_settings['distributions']
        self.excitation_threshold = calculation_settings['excitation_treshold']
        self.euler_angles_convention = calculation_settings['euler_angles_convention']
        self.fit_modulation_depth = calculation_settings['fit_modulation_depth']
        if self.fit_modulation_depth:
            self.interval_modulation_depth = calculation_settings['interval_modulation_depth']
            self.scale_range_modulation_depth = calculation_settings['scale_range_modulation_depth']