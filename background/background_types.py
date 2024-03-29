"""The list of supported background models."""
from background.background import Background
from background.exponential_background import ExponentialBackground
from background.stretched_exponential_background import StretchedExponentialBackground
from background.second_order_polynomial_background import SecondOrderPolynomialBackground
from background.third_order_polynomial_background import ThirdOrderPolynomialBackground
from background.fourth_order_polynomial_background import FourthOrderPolynomialBackground
from background.kellers_background import KellersBackground


background_types = {}
background_types["exp"] = ExponentialBackground
background_types["stretched_exp"] = StretchedExponentialBackground
background_types["polynom2"] = SecondOrderPolynomialBackground
background_types["polynom3"] = ThirdOrderPolynomialBackground
background_types["polynom4"] = FourthOrderPolynomialBackground
background_types["keller"] = KellersBackground