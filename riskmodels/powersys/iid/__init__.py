"""
These modules implement snapshot models for available conventional generation (ACG) and power surpluses (defined as available overall generation minus demand) in two-area systems. This means sequential time dependence is disregarded and instead an iid framework is assumed for all involved distributions. This is useful to compute risk indices for which time is integrated out, such as loss of load expectation (LOLE) and expected energy unserved (EEU). Available models include a general Monte Carlo model for arbitrary net demand and generation distributions, and an empirical model which also includes functionality to analyse risks under a 'share' policy in which shortfalls are shared across areas using interconnectors.
"""
