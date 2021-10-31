"""
These modules implement sequential models for available conventional generation (ACG) and power surpluses, defined as available overall generation minus demand. At the moment only empirical models for demand and intermittent generation are supported, given by vectors of historical observations, while the ACG models assumes generators are independent and are modeled as Markov Chains.
The modules are oriented toward large scale simulation for Monte Carlo estimation; this is done through Map Reduce processing patterns.
"""
