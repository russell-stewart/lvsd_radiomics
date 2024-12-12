from enum import Flag , auto
import numpy as np

class HFClassification(Flag):
    """
    Enumeration specifying possible etiology classes for HFrEF for the LVSD classifier project
    Field: Heart failure diagnosis
    Value: Defined as 2^n for combination through flags
    Use: Specify EXCLUDED or ICM or NICM. If NICM, you may join with an NICM subtype using the | operator
    """
    # Fallback for excluded patient
    EXCLUDED = 1
    # Definition of ICM
    ICM = 2
    # Definition of NICM
    NICM = 4
    # Definitions of etiologies underlying NICM
    DIPTHERIC = 8
    VIRAL = 16
    ALCOHOLIC = 32
    DRUG = 64
    PERIPARTUM = 128
    AMYLOID = 256
    METABOLIC = 512
    SARCOIDOSIS = 1024
    HYPERTENSIVE = 2048
    TUBERCULOUS = 4096
    TAKOTSUBO = 8192
    CHAGAS = 16384
    ARRHYTHMIA = 32768
    SYPHILIS = 65536
    HEMOCHROMATOSIS = 131072
    # Valvular heart disease
    VALVULAR = 262144


    def __str__(self):
        """
        Overrides the default `__str__` to print a pretty representation of bitwise OR'ed enums
        """
        member_names = [member.name for member in self.__class__ if member & self]
        return " | ".join(member_names)
    
    def __vector__(y):
        """
        Converts to a representation in a vector space with a dimension for each classification flag (ie. > 2^n , n E [1,18]).
        Indices in this vector are arranged such that `HFClassification(2**i)` is the label enum corresponding to that index.
        """
        vector_space = np.array([HFClassification(2**x) for x in range(1 , 19)])
        hot_indices = (vector_space & y).astype(bool).astype(int)

        return hot_indices
