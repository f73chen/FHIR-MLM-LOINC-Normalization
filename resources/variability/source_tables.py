"""Verbatim transcriptions of the published tables behind the perturbation register (thesis
Chapter 4 and Appendix B).

Nothing here is computed or decided. Each table holds the numbers as the source prints them,
including rows and columns that the analysis does not use. A reviewer who checks a page against
this file is then also checking the analysis.

analysis/a27_register_verification.py asserts that every value the register uses appears in the
corresponding table.

Ricos and Adeli are not repeated here, because their tables are already constants in
`variability_register.RECORDS` (pipeline/) and `a26.TABLE2` (analysis/). Friedberg's Table 5 is
here because the register derives its range %V from the table itself.
"""

# ------------------- Vu 2022, Hematol Rep 14(2):73-84, Table 5 -------------------
#: "CV mean of each instrument group for CBC EQA material". Read from a page image (p.81).
#: The first row of each level is the paper's "All methods CV (%)" row, which is the
#: between-method figure the register uses. The other rows are within-peer-group CVs. They
#: understate between-laboratory variation because they exclude between-method bias, and they
#: are here for contrast only.
VU2022_T5_COLUMNS = ("RBC", "Hb", "Hct", "MCH", "MCHC", "MCV", "RDW-CV", "WBC", "PLT", "MPV")
VU2022_T5 = {
    "Level 1 (N=78)": [
        ("All methods CV",                        None, (1.7, 1.5, 2.2, 1.8, 2.2, 1.7, 2.1, 2.8, 5.2, 4.1)),
        ("Sysmex XN Series",                        15, (1.5, 1.1, 2.1, 1.4, 1.9, 1.1, 1.1, 3.0, 4.5, 1.5)),
        ("Abbott Cell-Dyn Ruby",                    15, (2.0, 0.9, 2.3, 1.7, 1.8, 1.9, 2.3, 2.9, 5.3, 6.1)),
        ("Siemens/Bayer Advia 120/2120",            13, (1.6, 1.0, 1.7, 1.7, 1.9, 1.4, 1.1, 2.9, 5.4, 2.8)),
        ("Beckman Coulter DxH 600/800/900",         15, (1.0, 1.7, 1.2, 1.6, 1.8, 0.9, 1.4, 2.0, 4.7, 3.2)),
        ("Mindray BC 1000/2000/3000",               12, (2.4, 1.4, 2.7, 2.5, 3.6, 2.7, 2.1, 2.0, 3.5, 3.3)),
        ("Nihon Kohden Celtac Alpha",               13, (1.9, 3.0, 3.1, 1.9, 2.1, 2.1, 4.8, 4.2, 7.7, 7.6)),
    ],
    "Level 3 (N=75)": [
        ("All methods CV",                        None, (1.6, 0.7, 1.7, 1.6, 1.5, 0.8, 1.7, 2.5, 2.0, 4.2)),
        ("Sysmex XN Series",                        15, (1.7, 0.8, 1.7, 1.9, 1.5, 1.0, 2.1, 2.4, 2.4, 2.0)),
        ("Abbott Cell-Dyn Ruby",                    15, (1.1, 0.7, 1.2, 1.4, 1.1, 0.9, 2.3, 3.0, 2.2, 6.9)),
        ("Siemens/Bayer Advia 120/2120",            13, (2.2, 0.6, 2.4, 1.9, 2.3, 2.1, 2.2, 2.5, 1.6, 2.2)),
        ("Beckman Coulter DxH 600/800/900",         15, (0.8, 0.8, 1.0, 1.1, 1.0, 0.4, 1.1, 1.8, 2.5, 3.4)),
        ("Mindray BC 1000/2000/3000",               12, (1.1, 0.9, 1.0, 0.8, 0.8, 0.3, 0.7, 2.1, 1.6, 2.3)),
        ("Nihon Kohden Celtac Alpha",               13, (2.6, 0.4, 2.6, 2.6, 2.6, 0.2, 1.7, 3.3, 2.0, 8.3)),
    ],
}

# --------------- Meijer 2018, Clin Chem Lab Med 56(10):1698-1703 ---------------
#: Read from page images (pp.1700-1701). ~600 participants per sample, 10 European
#: countries, samples prepared by UK NEQAS BC, INR range 2.0-4.0.
#: Table 1: INR mean (CV%) per reagent, reagents with >=10 participants.
MEIJER2018_T1 = [
    ("Owren", "Owrens PT (Medirox)",                59, ((2.05, 4.0), (2.56, 5.1), (3.06, 4.9), (3.26, 5.5))),
    ("Owren", "Nycotest PT (Nycomed)",              50, ((2.10, 5.7), (2.57, 7.8), (3.07, 8.1), (3.27, 7.3))),
    ("Owren", "SPA 20/50 (Stago)",                  99, ((2.04, 5.9), (2.54, 6.7), (2.90, 6.9), (3.22, 7.8))),
    ("Owren", "Hepato Quick (Stago)",               26, ((2.08, 2.4), (2.60, 2.8), (3.03, 2.4), (3.30, 2.7))),
    ("Quick", "PT-Fib-Recomb (IL)",                 16, ((2.47, 9.3), (2.96, 9.8), (3.59, 21.2), (3.80, 11.3))),
    ("Quick", "Recombiplastin (IL/Hemoliance)",     40, ((2.25, 6.2), (2.65, 6.4), (3.12, 7.4), (3.49, 7.4))),
    ("Quick", "Innovin (Siemens)",                 104, ((2.11, 5.2), (2.39, 5.0), (3.09, 6.5), (3.11, 6.1))),
    ("Quick", "Thromborel S (Siemens)",             55, ((2.16, 6.5), (2.55, 6.7), (3.03, 6.9), (3.29, 7.3))),
    ("Quick", "Neoplastin CI (Stago)",              38, ((2.21, 4.3), (2.58, 5.4), (3.30, 4.8), (3.29, 4.9))),
    ("Quick", "Neoplastin CI Plus (Stago)",         74, ((2.15, 3.4), (2.62, 3.8), (3.25, 4.9), (3.33, 4.5))),
    ("Quick", "Neoplastin R (Stago)",               10, ((2.20, 5.0), (2.61, 6.1), (3.22, 7.8), (3.48, 4.6))),
]
#: Table 2: between-laboratory CV% by reagent principle. (n_quick, n_owren, cv_quick, cv_owren)
MEIJER2018_T2 = [(362, 243, 6.9, 5.8), (363, 240, 8.2, 6.7),
                 (354, 245, 8.9, 7.0), (357, 245, 8.5, 7.4)]
#: Table 3: with and without local INR calibration, pooled. (n_no, n_yes, cv_no, cv_yes)
MEIJER2018_T3 = [(264, 333, 7.8, 5.8), (266, 329, 8.5, 6.8),
                 (256, 335, 8.9, 7.6), (261, 333, 9.0, 6.8)]

# --------------- Paal 2021, Biochem Med 31(3):030902, Table 1 ---------------
#: Read from the PDF text layer. The paper labels the column CVA. The authors state that it
#: estimates analytical variation in an inter-laboratory setting. That is the quantity the
#: register needs, although the label differs from CV_BL.
PAAL2021_T1 = [
    ("Creatinine",  "umol/L",    "115 - 478",   168.0,  4.7,  4.5, 645),
    ("Calcium",     "mmol/L",    "1.81 - 3.35",   2.20, 3.1,  3.3, 615),
    ("Aldosterone", "nmol/L",    "0.3 - 20.1",    0.8, 18.6, 36.6, 203),
    ("PSA",         "ug/L",      "0 - 14.0",      0.4, 18.8,  6.8, 804),
    ("HbA1c",       "mmol/mol",  "36.0 - 71.2",  51.5,  5.1,  1.7, 810),
]

# --------------- Cwiklinska 2020, Biochem Med 30(2):020707 ---------------
#: 1,400 results, 183 laboratories, 10 surveys 2017-2019. Median inter-laboratory CV.
#: The manual ranges in the abstract (53-71% WBC, 55-70% RBC) are confirmed from the PDF text
#: layer. The per-method breakdown below comes from the publisher's HTML and has not been
#: re-read from the PDF. It is consistent with the ranges in the abstract.
CWIKLINSKA2020 = {
    "manual": [("Coverslip method", 71, "49-81", 70, "51-89"),
               ("Chamber counting from sediment", 62, None, 62, None),
               ("Direct chamber counting", 53, "47-72", 55, "43-74")],
    "automated": [("Analyser A", 14, "8-18", 16, None), ("Analyser B", 18, None, 10, "6-11"),
                  ("Analyser C", 33, "27-56", 39, "35-58"), ("Analyser D", 24, None, 22, None)],
}

# --------------- Cunningham 2012, Arch Pathol Lab Med 136(7):789-795 ---------------
#: CAP 2008 proficiency testing programme in coagulation. >3,500 laboratories, 50
#: fibrinogen methods. Read from the PDF abstract.
CUNNINGHAM2012 = {
    "all_method_cv_pct": 7.7, "method_specific_cv_range": "0.7-25.8",
    "all_method_bias_pct": 8.3, "method_specific_bias_range": "0.0-27.0",
    "pt_based_vs_clauss_pct": 11.6, "mechanical_vs_photooptical_cv_lower_pct": 46,
    "n_labs": 3500, "n_methods": 50,
}

# --------------- Friedberg 2007, Arch Pathol Lab Med 131:348-357, Table 5 ---------------
#: "Variation in Reference Intervals of Selected Analytes Among 163 Laboratories" (p.352), the adult
#: rows. For each reference limit, the seven columns give its spread over the participating
#: laboratories: minimum, 10th, 25th, 50th (median), 75th and 90th percentiles, maximum. Transcribed
#: from the PDF and checked against its text layer. n_labs is the count for the lower limit. For the
#: upper limit it is 161 for calcium, 155 for TSH, 161 for platelets and 159 for aPTT. Keys are LOINC
#: codes, and "718-7F" is the female row of haemoglobin (718-7, Hgb Bld-mCnc). The register derives
#: Friedberg's %V, (p90 - p10) / p50, from this grid. analysis/a27_register_verification.py
#: re-derives and checks it.
FRIEDBERG2007_T5_PCTS = (0, 10, 25, 50, 75, 90, 100)
FRIEDBERG2007_T5 = {
    # loinc:   (analyte label,             unit,     n_labs, lower limits,  upper limits)
    # 2823-3 Potassium SerPl-sCnc
    "2823-3":  ("Potassium, adult",        "mmol/L", 162, (3.0, 3.3, 3.5, 3.5, 3.6, 3.6, 4.1),
                                                          (4.5, 5.0, 5.0, 5.1, 5.2, 5.3, 5.7)),
    # 17861-6 Calcium SerPl-mCnc
    "17861-6": ("Calcium (total), adult",  "mg/dL",  162, (7.6, 8.3, 8.4, 8.5, 8.5, 8.8, 9.0),
                                                          (9.6, 10.1, 10.1, 10.2, 10.5, 10.5, 10.8)),
    # 2601-3 Magnesium SerPl-sCnc
    "2601-3":  ("Magnesium, adult",        "mg/dL",  157, (1.3, 1.5, 1.6, 1.6, 1.8, 1.8, 2.1),
                                                          (2.0, 2.2, 2.3, 2.4, 2.6, 2.7, 3.2)),
    # 3016-3 TSH SerPl-aCnc
    "3016-3":  ("TSH, adult",              "mIU/L",  154, (0.10, 0.30, 0.34, 0.35, 0.40, 0.47, 0.50),
                                                          (3.00, 4.20, 4.68, 4.94, 5.50, 5.60, 6.00)),
    # 718-7 Hgb Bld-mCnc
    "718-7":   ("Haemoglobin (male), adult", "g/dL", 161, (11.5, 13.0, 13.0, 13.6, 14.0, 14.0, 14.8),
                                                          (15.0, 16.5, 17.0, 17.5, 18.0, 18.0, 18.1)),
    "718-7F":  ("Haemoglobin (female), adult", "g/dL", 159, (10.4, 11.5, 11.7, 12.0, 12.0, 12.1, 14.0),
                                                            (13.5, 15.0, 15.5, 16.0, 16.0, 16.0, 18.0)),
    # 777-3 Platelet # Bld Auto
    "777-3":   ("Platelets, adult",        "10^9/L", 162, (100, 130, 130, 142, 150, 150, 174),
                                                          (316, 400, 400, 400, 440, 450, 500)),
    # 3173-2 aPTT Bld
    "3173-2":  ("aPTT, adult",             "s",      156, (17, 21, 22, 23, 25, 26, 32),
                                                          (28, 32, 33, 35, 37, 38, 43)),
}
