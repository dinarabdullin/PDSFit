import sys
import time
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.simulator import Simulator
from mathematics.random_samples_from_distribution import random_samples_from_distribution
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.histogram import histogram
from mathematics.chi2 import chi2
from mathematics.fft import fft
from supplement.definitions import const


class MonteCarloSimulator(Simulator):
    """Monte-Carlo simulation of PDS time traces."""
    
    def __init__(self):
        super().__init__()
        self.intrinsic_parameter_names = {
            "number_of_montecarlo_samples": "int",
            "excitation_threshold": "float",
            "euler_angles_convention": "str"
        }
        self.freq_inc_epr_spectrum = 0.001 # in GHz
        self.freq_inc_dipolar_spectrum = {} # in MHz
        self.minimum_r = 1.5 # minimal distance in nm
        self.field_orientations = []
        self.eff_gs_spinA = []
        self.det_probs_spinA = {}
        self.pump_probs_spinA = {}      
    
    
    def set_intrinsic_parameters(self, intrinsic_parameters):
        """Set calculation settings."""
        self.num_samples = intrinsic_parameters["number_of_montecarlo_samples"]
        self.distribution_types = intrinsic_parameters["distribution_types"]
        self.excitation_threshold = intrinsic_parameters["excitation_threshold"]
        self.euler_angles_convention = intrinsic_parameters["euler_angles_convention"]
    
        
    def set_background_model(self, background_model):
        """Set the background model and parameters."""
        self.background_model = background_model
    
    
    def run_precalculations(self, experiments, spins):
        """Computes the detection and pump probabilities for spin A."""
        # Random orientations of the magnetic field
        self.field_orientations = self.set_field_orientations()
        # Plot orientations of the magnetic field
        # rho_values, xi_values, phi_values = cartesian2spherical(self.field_orientations)
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, [], [], [], [], "magnetic_field_orientations.png")
        # Orientations of the applied static magnetic field in the frame of spin A
        field_orientations_spinA = self.field_orientations 
        for experiment in experiments:
            self.freq_inc_dipolar_spectrum[experiment.name] = 0.5 / np.amax(experiment.t)
            # Resonance frequencies and effective g-values of spin A
            res_freqs_spinA, self.eff_gs_spinA = spins[0].res_freq(field_orientations_spinA, experiment.magnetic_field)
            # Detection probabilities
            self.det_probs_spinA[experiment.name] = experiment.detection_probability(res_freqs_spinA)
            # Pump probabilities
            if experiment.technique == "peldor": 
                self.pump_probs_spinA[experiment.name] = experiment.pump_probability(res_freqs_spinA) 
            elif experiment.technique == "ridme":
                self.pump_probs_spinA[experiment.name] = experiment.pump_probability(
                    spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, self.eff_gs_spinA
                    )
     
    def set_field_orientations(self): 
        """Generate the random orientations of the magnetic field."""
        rho = np.ones(self.num_samples)
        pol = np.arccos(2 * np.random.random_sample(self.num_samples) - 1)
        azi = 2 * np.pi * np.random.random_sample(self.num_samples)
        return spherical2cartesian(rho, pol, azi)
    
    
    def simulate_epr_spectra(self, spins, experiments):
        """Compute the EPR spectrum of a spin system at several magnetic fields."""
        epr_spectra = []
        for experiment in experiments:
            epr_spectrum = self.simulate_epr_spectrum(spins, experiment.magnetic_field)
            epr_spectra.append(epr_spectrum)
        return epr_spectra
    
    
    def simulate_epr_spectrum(self, spins, field_value):
        """Compute the EPR spectrum of a spin system at one magnetic field."""
        if len(self.field_orientations) == 0:
            self.field_orientations = self.set_field_orientations()
        res_freqs_spinA, _ = spins[0].res_freq(self.field_orientations, field_value)
        weights_spinA = np.tile(spins[0].int_res_freq, (self.field_orientations.shape[0], 1))
        res_freqs_spinB, _ = spins[1].res_freq(self.field_orientations, field_value)
        weights_spinB = np.tile(spins[1].int_res_freq, (self.field_orientations.shape[0], 1))
        res_freqs = np.concatenate((res_freqs_spinA, res_freqs_spinB), axis = None)
        weights = np.concatenate((weights_spinA, weights_spinB), axis = None)
        min_freq, max_freq = np.amin(res_freqs) - 0.100, np.amax(res_freqs) + 0.100
        freq_grid = np.arange(min_freq, max_freq + self.freq_inc_epr_spectrum, self.freq_inc_epr_spectrum)
        spectrum = {
            "freq": freq_grid,
            "prob": histogram(res_freqs, bins = freq_grid, weights = weights)
            }      
        return spectrum
    
    
    def simulate_bandwidths(self, experiments):
        """Compute the bandwidths of detection and pump pulses."""
        bandwidths = []
        for experiment in experiments:
            if experiment.technique == "peldor":
                bandwidths_single_experiment = {
                    "detection_bandwidth": experiment.get_detection_bandwidth(),
                    "pump_bandwidth": experiment.get_pump_bandwidth()
                }
                bandwidths.append(bandwidths_single_experiment)
            elif experiment.technique == "ridme":
                bandwidths_single_experiment = {
                    "detection_bandwidth": experiment.get_detection_bandwidth()
                }
                bandwidths.append(bandwidths_single_experiment)
        return bandwidths   
    
    
    def simulate_time_traces(
        self, model_parameters, experiments, spins,
        more_output = [], reset_field_orientations = False, display_messages = True
        ): 
        """ Simulate PDS time traces for a given geometric model."""
        if display_messages:
            sys.stdout.write(
                "\n########################################################################\
                \n#                              Simulation                              #\
                \n########################################################################\n"
                )
        if more_output:
            simulated_time_traces = []
            simulated_data = {}
            for key in more_output:
                simulated_data[key] = []
            for experiment in experiments:
                simulated_time_trace, simulated_data_single_exp = self.simulate_time_trace(
                    model_parameters, experiment, spins, more_output, reset_field_orientations, display_messages
                    ) 
                simulated_time_traces.append(simulated_time_trace)
                for key in more_output:
                    simulated_data[key].append(simulated_data_single_exp[key])
            return simulated_time_traces, simulated_data
        else: 
            simulated_time_traces = []
            for experiment in experiments:
                simulated_time_trace = self.simulate_time_trace(
                    model_parameters, experiment, spins, more_output, reset_field_orientations, display_messages
                    ) 
                simulated_time_traces.append(simulated_time_trace)
            return simulated_time_traces
            

    def simulate_time_trace(
        self, model_parameters, experiment, spins,  
        more_output = [], reset_field_orientations = False, display_messages = False
        ):
        """Simulate a PDS time trace for a given geometric model."""
        if display_messages:
            sys.stdout.write("\nSimulating the PDS time trace of experiment \'{0}\'... ".format(experiment.name))
            sys.stdout.flush()
        if more_output:
            simulated_time_trace, simulated_data = self.compute_time_trace(
                model_parameters, experiment, spins, more_output, reset_field_orientations, display_messages = False
                )
            if display_messages:
                sys.stdout.write("done!\n") 
                sys.stdout.write("Background parameters:\n") 
                for parameter_name in self.background_model.parameter_full_names:
                    sys.stdout.write(
                        "{0}: {1:<15.6f}\n".format(self.background_model.parameter_full_names[parameter_name], simulated_data["background_parameters"][parameter_name])
                        )
                chi2_value = chi2(simulated_time_trace, experiment.s, experiment.noise_std)
                sys.stdout.write("Chi-squared: {0:<15.1f}\n".format(chi2_value))
                sys.stdout.flush()                
            return simulated_time_trace, simulated_data
        else:
            simulated_time_trace = self.compute_time_trace(
                model_parameters, experiment, spins, more_output, reset_field_orientations, display_messages = False
                ) 
            if display_messages:
                sys.stdout.write("done!\n")
                chi2_value = chi2(simulated_time_trace, experiment.s, experiment.noise_std)
                sys.stdout.write("Chi-squared: {0:<15.1f}\n".format(chi2_value))
                sys.stdout.flush()
            return simulated_time_trace


    def compute_time_trace(
        self, model_parameters, experiment, spins,  
        more_output = [], reset_field_orientations = False, display_messages = False
        ):
        """Computes a PDS time trace for a given geometric model."""
        timings = ["Timings:"]
        time_start = time.time()
        # Random orientations of the applied static magnetic field in the reference frame
        if reset_field_orientations:
            self.field_orientations = []
            self.eff_gs_spinA = []
            self.det_probs_spinA = {}
            self.pump_probs_spinA = {}
        if len(self.field_orientations) == 0:
            self.field_orientations = self.set_field_orientations()
        # Plot orientations of the magnetic field
        # rho_values, pol_values, azi_values = cartesian2spherical(self.field_orientations)
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], pol_values, azi_values, [], [], [], [], "magnetic_field_orientations.png")
        num_samples_per_mode = self.samples_per_mode(len(model_parameters["r_mean"]), model_parameters["rel_prob"])
        # Distance values
        r_values, num_samples_per_mode = self.set_r_values(
            model_parameters["r_mean"], 
            model_parameters["r_width"], 
            num_samples_per_mode
            )
        # Orientations of the distance vector in the CS of spin A (reference CS)
        r_orientations = self.set_r_orientations(
            model_parameters["xi_mean"],
            model_parameters["xi_width"], 
            model_parameters["phi_mean"],
            model_parameters["phi_width"],
            num_samples_per_mode
            )
        # Rotation matrices transforming the CS of spin A (reference CS) to CS of spin B
        spin_frame_rotations = self.set_spin_frame_rotations(
            model_parameters["alpha_mean"],
            model_parameters["alpha_width"],
            model_parameters["beta_mean"],
            model_parameters["beta_width"],
            model_parameters["gamma_mean"],
            model_parameters["gamma_width"],
            num_samples_per_mode
            )
        # Exchange coupling values
        j_values = self.set_j_values(
            model_parameters["j_mean"],
            model_parameters["j_width"],
            num_samples_per_mode
            )                                                                                               
        # Plot Monte-Carlo points
        # rho_values, xi_values, phi_values = cartesian2spherical(r_orientations)
        # euler_angles = spin_frame_rotations.inv().as_euler(self.euler_angles_convention, degrees=False)
        # alpha_values, beta_values, gamma_values = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points(r_values, xi_values, phi_values, alpha_values, beta_values, gamma_values, j_values)
        # Orientations of the magnetic field in both CSs
        field_orientations_spinA = self.field_orientations
        field_orientations_spinB = spin_frame_rotations.apply(self.field_orientations)
        timings.append("Monte-Carlo samples: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        time_start = time.time()
        
        # Resonance frequencies and effective g-values of both spins
        if len(self.eff_gs_spinA) == 0:
            res_freqs_spinA, eff_gs_spinA = spins[0].res_freq(field_orientations_spinA, experiment.magnetic_field)
        else:
            eff_gs_spinA = self.eff_gs_spinA
        res_freqs_spinB, eff_gs_spinB = spins[1].res_freq(field_orientations_spinB, experiment.magnetic_field)
        timings.append("Resonance frequencies: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        
        time_start = time.time()
        # Detection probabilities
        if self.det_probs_spinA == {}:
            det_probs_spinA = experiment.detection_probability(res_freqs_spinA)
        else:
            det_probs_spinA = self.det_probs_spinA[experiment.name]
        det_probs_spinB = experiment.detection_probability(res_freqs_spinB)
        # Pump probabilities
        if experiment.technique == "peldor": 
            if self.pump_probs_spinA == {}:
                pump_probs_spinA = experiment.pump_probability(res_freqs_spinA)
            else:
                pump_probs_spinA = self.pump_probs_spinA[experiment.name]
            pump_probs_spinB = experiment.pump_probability(res_freqs_spinB)
        elif experiment.technique == "ridme":
            if self.pump_probs_spinA == {}:
                pump_probs_spinA = experiment.pump_probability(
                    spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, eff_gs_spinA
                    )
            else:
                pump_probs_spinA = self.pump_probs_spinA[experiment.name]
            if spins[0].num_res_freq > 1:
                pump_probs_spinA = np.tile(pump_probs_spinA, spins[0].num_res_freq)
            pump_probs_spinB = experiment.pump_probability(
                spins[1].T1, spins[1].g_anisotropy_in_dipolar_coupling, eff_gs_spinB
                )  
            if spins[1].num_res_freq > 1:
                pump_probs_spinB = np.tile(pump_probs_spinB, spins[1].num_res_freq)
        # Take into account the weights of individual resonance frequencies and
        # the overlap of the pump and detection bandwidths
        weights_spinA = np.expand_dims(spins[0].int_res_freq, -1)
        weights_spinB = np.expand_dims(spins[1].int_res_freq, -1)
        corrected_det_probs_spinA = np.dot(det_probs_spinA * (1.0 - pump_probs_spinA), weights_spinA)
        corrected_det_probs_spinB = np.dot(det_probs_spinB * (1.0 - pump_probs_spinB), weights_spinB)
        corrected_pump_probs_spinA = np.dot(pump_probs_spinA * (1.0 - det_probs_spinA), weights_spinA)
        corrected_pump_probs_spinB = np.dot(pump_probs_spinB * (1.0 - det_probs_spinB), weights_spinB)
        timings.append("Detection/pump probabilities: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        time_start = time.time()
        
        # Modulation depths                                     
        modulation_depths = 1.0 / np.sum(corrected_det_probs_spinA + corrected_det_probs_spinB) * \
            (corrected_det_probs_spinA * corrected_pump_probs_spinB + corrected_det_probs_spinB * corrected_pump_probs_spinA)            
        modulation_depths = np.squeeze(modulation_depths, axis = 1)
        indices_nonzero_probs = np.where(modulation_depths > self.excitation_threshold)[0]
        modulation_depths = modulation_depths[indices_nonzero_probs]
        timings.append("Modulation depths: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        time_start = time.time()
        
        # Modulation frequencies
        field_orientations = self.field_orientations[indices_nonzero_probs]
        r_values = r_values[indices_nonzero_probs]
        r_orientations = r_orientations[indices_nonzero_probs]
        eff_gs_spinA = eff_gs_spinA[indices_nonzero_probs]
        eff_gs_spinB = eff_gs_spinB[indices_nonzero_probs]
        j_values = j_values[indices_nonzero_probs] 
        if not spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1)**2
        elif spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
            quant_axes_spinA = spins[0].quantization_axis(field_orientations, eff_gs_spinA)
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * quant_axes_spinA, axis=1) * \
                np.sum(r_orientations * field_orientations, axis=1)
        elif not spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            field_orientations_spinB = field_orientations_spinB[indices_nonzero_probs]
            spin_frame_rotations = spin_frame_rotations[indices_nonzero_probs]
            r_orientations_spinB = spin_frame_rotations.apply(r_orientations)
            quant_axes_spinB = spins[1].quantization_axis(field_orientations_spinB, eff_gs_spinB)
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1) * \
                np.sum(r_orientations_spinB * quant_axes_spinB, axis=1) 
        elif spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            quant_axes_spinA = spins[0].quantization_axis(field_orientations, eff_gs_spinA)
            field_orientations_spinB = field_orientations_spinB[indices_nonzero_probs]
            spin_frame_rotations = spin_frame_rotations[indices_nonzero_probs]
            r_orientations_spinB = spin_frame_rotations.apply(r_orientations)
            quant_axes_spinB = spins[1].quantization_axis(field_orientations_spinB, eff_gs_spinB)
            quant_axes_spinB_ref = spin_frame_rotations.inv().apply(quant_axes_spinB)
            angular_term = np.sum(quant_axes_spinA * quant_axes_spinB_ref, axis=1) - \
                3.0 * np.sum(r_orientations * quant_axes_spinA, axis=1) * np.sum(r_orientations_spinB * quant_axes_spinB, axis=1)
        dipolar_frequencies = const["Fdd"] * eff_gs_spinA * eff_gs_spinB * angular_term / r_values**3
        modulation_frequencies = dipolar_frequencies + j_values
        timings.append("Modulation frequencies: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        time_start = time.time()
        
        # PDS time trace
        form_factor = self.form_factor_from_dipolar_spectrum(
            experiment.t, modulation_frequencies, modulation_depths, self.freq_inc_dipolar_spectrum[experiment.name]
            )
        background_parameters = self.background_model.optimize_parameters(experiment.t, experiment.s, form_factor)
        simulated_time_trace = self.background_model.get_fit(experiment.t, background_parameters, form_factor)
        simulated_time_trace = simulated_time_trace / np.amax(simulated_time_trace) 
        timings.append("PDS time trace: {0}".format(datetime.timedelta(seconds = time.time() - time_start)))
        
        if display_messages:
            sys.stdout.write(
                "\nNumber of Monte-Carlo samples with non-zero weights: {0} out of {1}\n".format(indices_nonzero_probs.size, self.num_samples)
                )
            for instance in timings:
                sys.stdout.write("{0}\n".format(instance))
            sys.stdout.flush()
        
        if more_output:
            simulated_data = {}
            # Modulation depth
            if "modulation_depth" in more_output:
                total_modulation_depth = np.sum(modulation_depths)
                simulated_data["modulation_depth"] = total_modulation_depth
            # Background parameters
            if "background_parameters" in more_output: 
                simulated_data["background_parameters"] = background_parameters
            # Background
            if "background" in more_output: 
                if not "modulation_depth" in more_output:
                    total_modulation_depth = np.sum(modulation_depths)
                background = self.background_model.get_background(experiment.t, background_parameters, total_modulation_depth)
                simulated_data["background"] = background
            # Form factor
            if "form_factor" in more_output: 
                if not "background" in more_output:
                    if not "modulation_depth" in more_output:
                        total_modulation_depth = np.sum(modulation_depths)
                    background = self.background_model.get_background(experiment.t, background_parameters, total_modulation_depth)
                form_factor_sim = simulated_time_trace * np.amax(background) / background
                form_factor_exp = experiment.s * np.amax(background) / background
                simulated_data["form_factor"] = {"exp": form_factor_exp, "sim": form_factor_sim}
            # Dipolar spectrum
            if "dipolar_spectrum" in more_output: 
                if not "form_factor" in more_output:
                    if not "background" in more_output:
                        if not "modulation_depth" in more_output:
                            total_modulation_depth = np.sum(modulation_depths)
                        background = self.background_model.get_background(experiment.t, background_parameters, total_modulation_depth)
                    form_factor_sim = simulated_time_trace * np.amax(background) / background
                    form_factor_exp = experiment.s * np.amax(background) / background
                dc_offset = 1 - total_modulation_depth * background_parameters["scale_factor"]
                freq_grid, spectrum_sim = fft(experiment.t, form_factor_sim, dc_offset = dc_offset)
                _, spectrum_exp = fft(experiment.t, form_factor_exp, dc_offset = dc_offset)
                simulated_data["dipolar_spectrum"] = {"freq": freq_grid, "exp": spectrum_exp, "sim": spectrum_sim}
            # Distribution of dipolar angles
            if "dipolar_angle_distribution" in more_output: 
                theta = const["rad2deg"] * np.arccos(np.sum(field_orientations * r_orientations, axis=1))
                theta = np.where(theta < 0, -1 * theta, theta)
                theta_grid = np.arange(0.0, 181.0, 1.0) 
                theta_probs = histogram(theta, bins = theta_grid, weights = modulation_depths)
                theta_probs = const["rad2deg"] * theta_probs / sum(theta_probs)
                simulated_data["dipolar_angle_distribution"] = {"angle": theta_grid, "prob": theta_probs}
            return simulated_time_trace, simulated_data
        else:
            return simulated_time_trace
    
    
    def samples_per_mode(self, num_modes, rel_prob):
        """Calculate the number of samples per mode in multimodal distributions."""
        num_samples_per_mode = np.zeros(num_modes, dtype=int)
        if num_modes == 1:
            num_samples_per_mode[0] = self.num_samples
        else:
            for i in range(num_modes - 1):
                num_samples_per_mode[i] = int(self.num_samples * rel_prob[i])
            num_samples_per_mode[num_modes - 1] = self.num_samples - sum(num_samples_per_mode)
        return num_samples_per_mode
    
    
    def set_r_values(self, r_mean, r_width, num_samples_per_mode):
        """Generate random samples of r from the distribution P(r)."""
        r_values = []
        valid_indices = []
        valid_num_samples_per_mode = np.zeros(num_samples_per_mode.size, dtype=int)
        for i in range(len(num_samples_per_mode)):
            r_values_single_mode = random_samples_from_distribution(
                self.distribution_types["r"], r_mean[i], r_width[i], num_samples_per_mode[i], sine_weighted = False
            )
            # Check that all r values are above the lower limit
            # If not, record the number and indices of valid r samples
            valid_indices_single_mode = np.where(r_values_single_mode >= self.minimum_r)[0]
            valid_indices.append(valid_indices_single_mode)
            valid_num_samples_per_mode[i] = valid_indices_single_mode.size
            r_values.append(r_values_single_mode[valid_indices_single_mode])
        # If needed, replace invalid samples with new samples.
        if np.all(valid_num_samples_per_mode == num_samples_per_mode):
            r_values = np.concatenate(r_values, axis = None)
            return r_values, num_samples_per_mode
        else:
            # Update the relative weigths.
            valid_num_samples = np.sum(valid_num_samples_per_mode)
            new_rel_prob = valid_num_samples_per_mode.astype("float") / float(valid_num_samples)
            new_num_samples_per_mode = self.samples_per_mode(len(r_mean), new_rel_prob)
            # Add missing samples.
            new_r_values = []
            for i in range(len(new_num_samples_per_mode)):
                num_missing_samples = new_num_samples_per_mode[i] - valid_num_samples_per_mode[i]
                if num_missing_samples <= 0:
                    new_r_values.append(r_values[:new_num_samples_per_mode[i]])
                else:
                    r_values_single_mode = r_values[i]
                    for _ in range(num_missing_samples):
                        r_value = 0.0
                        while r_value < self.minimum_r:
                            r_value = random_samples_from_distribution(
                                self.distribution_types["r"], r_mean[i], r_width[i], 1, sine_weighted = False
                                )
                        r_values_single_mode = np.append(r_values_single_mode, r_value)
                    new_r_values.append(r_values_single_mode)
            new_r_values = np.concatenate(new_r_values, axis = None)
            return new_r_values, new_num_samples_per_mode


    def set_r_orientations(
        self, xi_mean, xi_width, phi_mean, phi_width, num_samples_per_mode
        ):
        """Generate random samples of xi and phi from distributions P(xi)*sin(xi) and P(phi).
        Compute the orientations of the distance vector in the reference frame."""
        xi_values, phi_values = [], []
        for i in range(len(num_samples_per_mode)):
            xi_values_single_mode = random_samples_from_distribution(
                self.distribution_types["xi"], xi_mean[i], xi_width[i], num_samples_per_mode[i], sine_weighted = True
                )
            phi_values_single_mode = random_samples_from_distribution(
                self.distribution_types["phi"], phi_mean[i], phi_width[i], num_samples_per_mode[i], sine_weighted = False
                )
            xi_values.append(xi_values_single_mode)
            phi_values.append(phi_values_single_mode)
        xi_values = np.concatenate(xi_values, axis = None)
        phi_values = np.concatenate(phi_values, axis = None)
        r_orientations = spherical2cartesian(np.ones(self.num_samples), xi_values, phi_values)
        # Plot Monte-Carlo points
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, [], [], [], [], "spherical_coordinates.png")
        return r_orientations
    
    
    def set_spin_frame_rotations(
        self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, num_samples_per_mode
        ):
        """ Generate random samples of alpha, beta, and gamma from distributions P(alpha), P(beta)*sin(beta), and P(gamma).
        Compute the rotation matrices transforming the CS of spin A into the CS of spin B."""
        alpha_values, beta_values, gamma_values = [], [], []
        for i in range(len(num_samples_per_mode)):
            alpha_values_single_mode = random_samples_from_distribution(
                self.distribution_types["alpha"], alpha_mean[i], alpha_width[i], num_samples_per_mode[i], sine_weighted = False
                )
            beta_values_single_mode = random_samples_from_distribution(
                self.distribution_types["beta"], beta_mean[i], beta_width[i], num_samples_per_mode[i], sine_weighted = True
                )
            gamma_values_single_mode = random_samples_from_distribution(
                self.distribution_types["gamma"], gamma_mean[i], gamma_width[i], num_samples_per_mode[i], sine_weighted = False
                )
            alpha_values.append(alpha_values_single_mode)
            beta_values.append(beta_values_single_mode)
            gamma_values.append(gamma_values_single_mode)
        alpha_values = np.concatenate(alpha_values, axis = None)
        beta_values = np.concatenate(beta_values, axis = None)
        gamma_values = np.concatenate(gamma_values, axis = None)
        spin_frame_rotations = Rotation.from_euler(
            self.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values))
            )
        # Convert active rotations to passive rotations
        spin_frame_rotations = spin_frame_rotations.inv()
        # Plot Monte-Carlo points
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], [], [], alpha_values, beta_values, gamma_values, [], "euler_angles.png")
        return spin_frame_rotations
    
    
    def set_j_values(self, j_mean, j_width, num_samples_per_mode):
        """Generate randon samples of J from distribution P(J)."""
        j_values = []
        for i in range(len(num_samples_per_mode)):
            j_values_single_mode = random_samples_from_distribution(
                self.distribution_types["j"], j_mean[i], j_width[i], num_samples_per_mode[i], sine_weighted = False
                )
            j_values.append(j_values_single_mode)
        j_values = np.concatenate(j_values, axis = None)
        return j_values
    
    
    def form_factor_from_dipolar_frequencies(self, t, f, d):
        """Compute the form factor based on the modulation frequencies and depths."""
        nt = t.size
        form_factor = np.ones(nt)
        if f.size != 0:
            for i in range(nt):
                form_factor[i] -= np.sum(d * (1 - np.cos(2 * np.pi * f * t[i])))
        return form_factor
    
    
    def form_factor_from_dipolar_spectrum(self, t, f, d, f_inc):
        """Compute the form factor based on the dipolar spectrum."""
        nt = t.size
        form_factor = np.ones(nt)
        if f.size != 0:
            f_min, f_max = np.amin(f), np.amax(f)
            if f_max - f_min > f_inc:
                new_f = np.arange(np.amin(f), np.amax(f), f_inc)
                new_d = histogram(f, bins = new_f, weights = d)
            else:
                new_f = np.array([f_min])
                new_d = np.array([np.sum(d)])
            for i in range(nt):
                form_factor[i] -= np.sum(new_d * (1 - np.cos(2 * np.pi * new_f * t[i])))
        return form_factor