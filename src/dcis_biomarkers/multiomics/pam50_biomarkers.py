"""
PAM50 and Oncotype DX Gene Signature Biomarkers for Breast DCIS Progression.
Contains standard clinical gene panels for breast cancer subtyping and risk scoring.
"""

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20",
    "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1",
    "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "MELK", "MIA", "MKI67", "MLPH",
    "MMP11", "MYC", "NAT1", "NDC80", "NUF2", "PGR", "PHGDH", "PTTG1", "RAB27B",
    "RPA3", "TYMS", "UBE2C", "UBE2T"
]

KEY_DRIVERS = [
    "TP53", "PIK3CA", "MAP3K1", "KMT2C", "CDH1", "PTEN", "AKT1", "GATA3", "RB1",
    "ATM", "BRCA1", "BRCA2", "CHEK2", "PALB2"
]

ALL_BIOMARKER_GENES = list(set(PAM50_GENES + KEY_DRIVERS))
