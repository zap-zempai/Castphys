
from enum import Enum
from pathlib import Path

import pandas as pd


class Gender(Enum):
    MALE = 0
    FEMALE = 1
    OTHERS = 2
    PREFER_NOT_SAY = 3

GENDER_MAPPING = {
    "male": Gender.MALE,
    "female": Gender.FEMALE,
    "others": Gender.OTHERS,
    "prefer not to say": Gender.PREFER_NOT_SAY
}

class Ethnicity(Enum):
    """
    Asian / Pacific Islander
    African
    African American
    Hispanic /Latin
    Native American / Alaskan Native
    Caucasian
    Biracial / Multiracial
    Other
    I prefer not to answer
    """
    ASIAN_OR_PACIFIC_ISLANDER = 0
    AFRICAN = 1
    AFRICAN_AMERICAN = 2
    HISPANIC_OR_LATIN = 3
    NATIVE_AMERICAN_OR_ALASKAN_NATIVE = 4
    CAUCASIAN = 5
    BIRACIAL_OR_MULTIRACIAL = 6
    OTHER = 7
    PREFER_NOT_SAY = 8

EHTNICITY_MAPPING = {
    "Asian / Pacific Islander": Ethnicity.ASIAN_OR_PACIFIC_ISLANDER,
    "African": Ethnicity.AFRICAN,
    "African American": Ethnicity.AFRICAN_AMERICAN,
    "Hispanic / Latin": Ethnicity.HISPANIC_OR_LATIN,
    "Native American / Alaskan Native": Ethnicity.NATIVE_AMERICAN_OR_ALASKAN_NATIVE,
    "Caucasian": Ethnicity.CAUCASIAN,
    "Biracial / Multiracial": Ethnicity.BIRACIAL_OR_MULTIRACIAL,
    "Other": Ethnicity.OTHER,
    "Prefer not to say": Ethnicity.PREFER_NOT_SAY
}

class DemographicRecord:
    def __init__(self,
                 subject_id: int,
                 age: int,
                 gender: Gender,
                 ethinicity: Ethnicity,
                 is_tca: bool) -> None:
        self.subject_id = subject_id
        self.age = age
        self.gender = gender
        self.ethnicity = ethinicity
        self.is_tca = is_tca

class DemographicRecordSaver:
    @staticmethod
    def save(demographic_record: DemographicRecord,
             output_filename: Path):
        df = pd.DataFrame()
        df["subject_id"] = [demographic_record.subject_id]
        df["age"] = [demographic_record.age]
        df["gender"] = [demographic_record.gender.name]
        df["ethnicity"] = [demographic_record.ethnicity.name]
        df["is_tca"] = [demographic_record.is_tca]

        df.to_csv(output_filename, index=False)
