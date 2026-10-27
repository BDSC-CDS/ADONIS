from abc import ABC

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------

class AbstractVariable(ABC):
    def process(self, df):
        return df

class AdmissionDate(AbstractVariable):
    def process(self, df):
        # Convert adm_date from string to datetime (pandas handles the timezone automatically)
        df["adm_date"] = pd.to_datetime(
            df["adm_date"], format="mixed", utc=True
        )
        df["adm_date_rounded"] = df["adm_date"].dt.floor(freq="24h")

        # Convert dis_date from string to datetime (pandas handles the timezone automatically)
        df["disc_date"] = pd.to_datetime(
            df["disc_date"], format="mixed", utc=True
        )
        df["disc_date_rounded"] = df["disc_date"].dt.ceil(freq="24h")

        # Drop duplicates
        df = df.drop_duplicates(
            subset=[
                "stay_id",
                "adm_date",
                "disc_date",
            ]
        ).copy()

        ## Length of stay
        los = df["disc_date"] - df["adm_date"]

        # Calculate the length of stay
        df["length_stay"] = los.dt.days + 1

        # Calculate the length of stay in hours
        df["length_stay_hours"] = los.dt.total_seconds() / 3600

        return df

class BirthDate(AbstractVariable):
    def process(self, df):
        # Convert birth_date from string to datetime with the appropriate format
        df["birth_date"] = pd.to_datetime(
            df["birth_date"], format="%d.%m.%Y", utc=False
        )
        df["birth_date"] = df["birth_date"].dt.tz_localize("Europe/Zurich")

        return df

class BirthWeight(AbstractVariable):
    def process(self, df):
        # Convert variables from string to numerical
        df["birth_weight"] = pd.to_numeric(df["birth_weight"])

        df.loc[df["birth_weight"] < 10, "birth_weight"] = (
            1000 * df[df["birth_weight"] < 10]["birth_weight"]
        )

        return df

class Death(AbstractVariable):
    def process(self, df):
        # Convert death_date from string to datetime (pandas handles the timezone automatically)
        df["death_date"] = pd.to_datetime(
            df["death_date"], format="mixed", utc=True
        )

        # Convert death_status from string to int64
        df = df.rename(columns={"death_status": "death"})
        df["death"] = pd.to_numeric(df["death"])

        return df

class GestationalAge(AbstractVariable):
    def process(self, df):
        df = df.drop_duplicates(
            subset=["patient_id"]
        ).copy()

        # Convert variables from string to numerical
        df["ga_days"] = pd.to_numeric(df["ga_days"])
        df["ga_weeks"] = pd.to_numeric(df["ga_weeks"])

        df = df.dropna(subset=["ga_weeks"])

        df["ga_days"] = df["ga_days"].astype(int)
        df["ga_weeks"] = np.floor(df["ga_weeks"])

        return df

class LengthOfStay(AbstractVariable):
    def process(self, df):
        df["start_date"] = pd.to_datetime(
            df["start_date"], format="mixed", utc=True
        )
        df["start_date"] = df["start_date"].dt.floor("min")

        df["end_date"] = pd.to_datetime(
            df["end_date"], format="mixed", utc=True
        )
        df["end_date"] = df["end_date"].dt.floor("min")

        # Drop duplicates
        df = df.drop_duplicates(
            subset=[
                "stay_id",
                "start_date",
                "end_date",
                "ward",
            ]
        )

        return df

class NEC(AbstractVariable):
    def process(self, df):
        df = df.drop_duplicates(
            subset=["stay_id"]
        ).copy()

        # Convert variables from string to numerical
        df["nec"] = pd.to_numeric(df["nec"])
        return df

class Sepsis(AbstractVariable):
    def __init__(self):
        self.infection_mapping = {
            "CLABSI": [
                "Infection of bloodstream co-occurrent and due to central venous catheter in situ (disorder)",
            ],
            "NEC": [
                "Infectious peritonitis (disorder)",
            ],
            "Pneumonia": [
                "Infective pneumonia (disorder)",
                "Neonatal pneumonia (disorder)",
            ],
            "UTI": [
                "Urinary tract infectious disease (disorder)",
            ],
            "CNS": [
                "Infective encephalitis (disorder)",
                "Neonatal meningitis (disorder)",
            ],
            "Other": [
                "Bacterial cellulitis (disorder)",
                "Infective arthritis (disorder)",
                "Infectious disease (disorder)",
                "Infective endocarditis (disorder)",
                "Local infection of wound (disorder)",
                "Osteomyelitis (disorder)",
            ]
        }

        self.organism_mapping = {
            "CoNS": [
                "Staphylococcus capitis (organism)",
                "Staphylococcus caprae (organism)",
                "Staphylococcus condimenti (organism)",
                "Staphylococcus epidermidis (organism)",
                "Staphylococcus haemolyticus (organism)",
                "Staphylococcus hominis (organism)",
                "Staphylococcus warneri (organism)",
                "Staphylococcus, coagulase negative (organism)",
            ],
            "Staphylococcus_aureus": [
                "Methicillin resistant Staphylococcus aureus (organism)",
                "Staphylococcus aureus (organism)",
            ],
            "Other_Gram-positive_bacteria": [
                "Aerococcus urinae (organism)",
                "Bacillus cereus (organism)",
                "Bifidobacterium breve (organism)",
                "Bifidobacterium longum (organism)",
                "Bifidobacterium longum (substance)",
                "Clostridioides difficile (organism)",
                "Clostridium neonatale (organism)",
                "Cutibacterium acnes (organism)",
                "Enterococcus faecalis (organism)",
                "Enterococcus faecium (organism)",
                "Genus Arthrobacter (organism)",
                "Genus Enterococcus (organism)",
                "Genus Streptococcus (organism)",
                "Gram-positive coccus (organism)",
                "Granulicatella adiacens (organism)",
                "Lacticaseibacillus rhamnosus (organism)",
                "Lactococcus lactis (organism)",
                "Limosilactobacillus fermentum (organism)",
                "Listeria monocytogenes (organism)",
                "Listeria monocytogenes serotype 4b (organism)",
                "Micrococcus luteus (organism)",
                "Peptoniphilus harei (organism)",
                "Staphylococcus, coagulase positive (organism)",
                "Streptococcus agalactiae (organism)",
                "Streptococcus anginosus (organism)",
                "Streptococcus anginosus group (organism)",
                "Streptococcus bovis group (organism)",
                "Streptococcus cristatus (organism)",
                "Streptococcus dysgalactiae (organism)",
                "Streptococcus equinus (organism)",
                "Streptococcus mitis (organism)",
                "Streptococcus mitis group (organism)",
                "Streptococcus pneumoniae (organism)",
                "Streptococcus pyogenes (organism)",
                "Streptococcus salivarius group (organism)",
            ],
            "Escherichia_coli": [
                "Escherichia coli (organism)",
                "Extended spectrum beta-lactamase producing Escherichia coli (organism)",
            ],
            "Klebsiella_spp": [
                "Klebsiella aerogenes (organism)",
                "Klebsiella oxytoca (organism)",
                "Klebsiella pneumoniae (organism)",
            ],
            "Pseudomonas_spp": [
                "Pseudomonas aeruginosa (organism)",
            ],
            "Other_Gram-negative_bacteria": [
                "Acinetobacter baumannii (organism)",
                "Bacteroides fragilis group (organism)",
                "Citrobacter freundii (organism)",
                "Citrobacter koseri (organism)",
                "Cronobacter sakazakii (organism)",
                "Elizabethkingia meningoseptica (organism)",
                "Enterobacter cancerogenus (organism)",
                "Enterobacter cloacae (organism)",
                "Enterobacter cloacae complex (organism)",
                "Genus Achromobacter (organism)",
                "Genus Neisseria (organism)",
                "Haemophilus haemolyticus (organism)",
                "Haemophilus influenzae (organism)",
                "Proteus mirabilis (organism)",
                "Serratia marcescens (organism)",
            ],
            "Fungi": [
                "Candida albicans (organism)",
                "Candida parapsilosis (organism)",
            ],
        }

        # Build reverse mapping: disorder string -> short key
        self.reverse_infection_mapping = {
            infection: key
            for key, infections in self.infection_mapping.items()
            for infection in infections
        }

        self.reverse_organism_mapping = {
            organism: key
            for key, organisms in self.organism_mapping.items()
            for organism in organisms
        }

    def process(self, df):
        # Convert variables from string to numerical
        df["sepsis"] = pd.to_numeric(df["sepsis"])
        df["sepsis_not_confirmed"] = pd.to_numeric(df["sepsis_not_confirmed"])
        df["sepsis_eos"] = pd.to_numeric(df["sepsis_eos"])
        df["sepsis_los"] = pd.to_numeric(df["sepsis_los"])
        df["sepsis_halos"] = pd.to_numeric(df["sepsis_halos"])
        df["sepsis_calos"] = pd.to_numeric(df["sepsis_calos"])

        df["datetime_hemoculture"] = pd.to_datetime(
            df["datetime_hemoculture"], format="mixed", utc=True
        )
        df["datetime_hemoculture_sepsis_not_confirmed"] = pd.to_datetime(
            df["datetime_hemoculture_sepsis_not_confirmed"], format="mixed", utc=True
        )

        # Split into sepsis and non-sepsis early
        mask_sepsis = (df["sepsis"] == 1) | (df["sepsis_not_confirmed"] == 1)
        df_no_sepsis = df[~mask_sepsis].drop(columns=["infection", "organism"])
        df = df[mask_sepsis].copy()

        # Convert focus of infection codes to short keys
        df["infection"] = df["infection"].map(self.reverse_infection_mapping)
        df.loc[df["infection"].isna(), "infection"] = "Primary_BSI"

        # Convert trigger organism codes to short keys
        df["organism"] = df["organism"].map(self.reverse_organism_mapping)

        # One-hot encode infections per key-group 
        infection_dummies = (
            pd.get_dummies(df.set_index(["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"])["infection"])
            .groupby(["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"], dropna=False)
            .max()
            .add_prefix("infection_")
        )

        # One-hot encode organisms per key-group 
        organism_dummies = (
            pd.get_dummies(df.set_index(["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"])["organism"])
            .groupby(["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"], dropna=False)
            .max()
            .add_prefix("organism_")
        )

        # Collapse sepsis rows to one row per key-group 
        df = (
            df.drop(columns=["infection", "organism"])
            .groupby(["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"], as_index=False, dropna=False)
            .first()
            .merge(infection_dummies, on=["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"])
            .merge(organism_dummies, on=["stay_id", "datetime_hemoculture", "datetime_hemoculture_sepsis_not_confirmed"])
        )

        # Ensure all expected columns exist (fill 0 for unseen categories)
        for cat in list(self.infection_mapping.keys()) + ["Primary BSI"]:
            col = f"infection_{cat.replace(' ', '_')}"
            if col not in df.columns:
                df[col] = 0

        for cat in self.organism_mapping.keys():
            col = f"organism_{cat.replace(' ', '_')}"
            if col not in df.columns:
                df[col] = 0

        # Remove "Other" infection when there is a more specific one
        infection_cols = [col for col in df.columns if col.startswith("infection_") and col != "infection_Other"]
        has_specific = df[infection_cols].any(axis=1)
        df.loc[has_specific, "infection_Other"] = 0

        # Remove "CoNS" organism when there is a more specific one
        organism_cols = [col for col in df.columns if col.startswith("organism_") and col != "organism_CoNS"]
        has_specific = df[organism_cols].any(axis=1)
        df.loc[has_specific, "organism_CoNS"] = 0

        # Add "Polymicrobial_infection" organism when more than one is present
        organism_cols = [col for col in df.columns if col.startswith("organism_")]
        df["organism_Polymicrobial_infection"] = (df[organism_cols].sum(axis=1) > 1).astype(int)

        # Re-attach non-sepsis rows (all infection/organism columns = 0) 
        df = pd.concat([df, df_no_sepsis], ignore_index=True)
        for col in df.columns:
            if col.startswith("infection_") or col.startswith("organism_"):
                df[col] = df[col].fillna(0).astype(int)

        # Drop duplicates, keeping row with highest sepsis_halos
        df = df.sort_values(["stay_id", "sepsis_halos"], ascending=[True, False])
        df = df.drop_duplicates(subset=["stay_id"])

        return df

class Sex(AbstractVariable):
    def process(self, df):
        # Convert variables from string to numerical
        df["sex"] = df["sex"].map({"F": 0, "M": 1})

        return df
    

demographics_dict = {
    "patient_info": AbstractVariable,
    "admission_date": AdmissionDate,
    "birth_date": BirthDate,
    "birth_weight": BirthWeight,
    "death": Death,
    "gestational_age": GestationalAge,
    "length_of_stay": LengthOfStay,
    "nec": NEC,
    "sepsis": Sepsis,
    "sex": Sex,
}