

solver_list= ["bdd", "msat", "yices", "cvc4", "picosat", "btor", "z3", "dummy"]
solver_enum = dict([(v, i) for i, v in enumerate(solver_list)])


supported_logic = {
    'bdd': [
        'QF_BOOL', 'BOOL'
    ],
    'msat': [
        'QF_ABV', 'QF_AUFBV*', 'QF_RDL', 'QF_AX', 'QF_UFIDL', 'QF_LRA',
        'QF_ALIA', 'QF_AUFBV', 'QF_BOOL', 'QF_UFLIRA', 'QF_UFLRA', 'QF_AX*',
        'QF_LIA', 'QF_AUFBVLIRA*', 'QF_ABV*', 'QF_BV', 'QF_IDL', 'QF_AUFBVLIRA',
        'QF_ALIA*', 'QF_UF', 'QF_UFBV', 'QF_AUFLIA*', 'QF_UFLIA', 'QF_AUFLIA'
    ],
    'yices': [
        'QF_LIA', 'QF_UFIDL', 'QF_LRA', 'QF_UFLRA', 'QF_UFBV', 'QF_BOOL',
        'QF_RDL', 'QF_BV', 'QF_IDL', 'QF_UFLIRA', 'QF_UF', 'QF_UFLIA'
    ],
    'cvc4': [
        'LIA', 'QF_AUFLIA', 'QF_RDL', 'QF_AX', 'QF_UFIDL', 'QF_LRA', 'QF_AUFBV',
        'QF_BOOL', 'QF_UFLIRA', 'QF_UF', 'QF_UFLIA', 'QF_LIA', 'QF_BV', 'QF_IDL',
        'QF_AUFBVLIRA', 'LRA', 'QF_UFLRA', 'QF_UFBV', 'UFLRA', 'QF_ABV', 'BOOL',
        'UFLIRA',
        'AUFLIRA'
    ],
    'picosat': [
        'QF_BOOL'
    ],
    'btor': [
        'QF_BV', 'QF_UFBV', 'QF_ABV', 'QF_AUFBV', 'QF_AX'
    ],
    'z3': [
        'QF_LIA', 'LIA', 'QF_AUFBV*', 'QF_AUFLIA*', 'QF_AUFBVLIRA*',
        'QF_ABV*', 'QF_AUFLIA', 'QF_BV', 'QF_IDL', 'QF_RDL', 'QF_AX', 'QF_ALIA*',
        'LRA', 'QF_UFBV', 'QF_UFIDL', 'QF_LRA', 'QF_UF', 'QF_ALIA', 'QF_AX*',
        'QF_NIA', 'QF_NRA', 'UFLRA', 'QF_ABV', 'QF_AUFBV', 'QF_AUFBVLIRA',
        'QF_BOOL', 'BOOL', 'QF_UFLIRA', 'QF_UFLRA', 'QF_UFLIA', 'UFLIRA',
        'AUFLIRA',
        'AUFNIRA'
    ],
    'dummy': [
        "ABVFP", "ALIA", "AUFBVDTLIA", "AUFDTLIA", "AUFLIA", "AUFLIRA", "AUFNIRA", 
        "BV", "BVFP", "FP", "LIA", "LRA", "NIA", "NRA", "QF_ABV", "QF_ABVFP", "QF_ALIA", 
        "QF_ANIA", "QF_AUFBV", "QF_AUFLIA", "QF_AUFNIA", "QF_AX", "QF_BV", "QF_BVFP", 
        "QF_DT", "QF_FP", "QF_IDL", "QF_LIA", "QF_LIRA", "QF_LRA", "QF_NIA", "QF_NIRA", 
        "QF_NRA", "QF_RDL", "QF_UF", "QF_UFBV", "QF_UFIDL", "QF_UFLIA", "QF_UFLRA", 
        "QF_UFNIA", "QF_UFNRA", "UF", "UFBV", "UFDT", "UFDTLIA", "UFIDL", "UFLIA", 
        "UFLRA", "UFNIA", ""
    ]
}

# patch supported logic #

supported_logic['bdd'] += []
#supported_logic['msat'] += ['QF_NIA']
#supported_logic['yices'] += ['QF_NIA', 'QF_ABV', 'QF_NRA']
#supported_logic['cvc4'] += ['BV', 'UFLIA', 'UFIDL', 'QF_NIA']
supported_logic['picosat'] += []
#supported_logic['btor'] += ['QF_UFLRA']
supported_logic['z3'] += ['BV', 'UFBV', 'UFLIA', 'UFIDL', 'ALIA', 'QF_UFNRA', 'NRA']

logic_enum = dict([(x, i) for i, x in enumerate(supported_logic['dummy'])])
