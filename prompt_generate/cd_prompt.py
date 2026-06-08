from __future__ import annotations

import json
from typing import Dict


class BaseObjTask:
    DATASET_NAME = ""
    SHORT_NAME_MAP: Dict[str, str] = {}
    DOMAIN_KNOWLEDGE = (
        "Prefer sparse direct causal edges. Avoid edges that are better "
        "explained by common causes, indirect paths, or missingness-induced correlations."
    )

    def get_shortname_map(self) -> Dict[str, str]:
        return dict(self.SHORT_NAME_MAP)

    def generate_instructor_scene(self) -> str:
        return self.DOMAIN_KNOWLEDGE

    def generate_prompt_qa(self, add_cot: bool = False) -> Dict[str, str]:
        short_map_json = json.dumps(self.get_shortname_map(), ensure_ascii=False, indent=2)
        prompt = f"""
Role:
You are a cautious causal discovery assistant. For each input triplet, judge
the most reliable direct directed causal edges inside that local triplet.

Dataset:
{self.DATASET_NAME}

Use the following evidence:
1. variable semantics and domain knowledge;
2. the missingness report, to avoid mistaking missingness-induced association for causation;
3. differential feedback from the previous round, when provided;
4. the triplet-query rule: only judge direct edges among the three variables in the current triplet.

Output contract:
Return only one JSON object. Do not return Markdown or explanations.
The JSON object must have exactly one top-level key, "results".
Each item in "results" must follow this schema:

{{
  "results": [
    {{
      "triple": ["X", "Y", "Z"],
      "key": "X|Y|Z",
      "edges": [["X", "Y"]]
    }}
  ]
}}

Strict rules:
- The number and order of results must match the input triplets.
- Each "key" is the triplet joined by "|".
- Each edge is [source, target].
- Use only short names appearing in the current triplet.
- Do not output self-loops, duplicate edges, or both directions for one pair.
- Return an empty edge list when evidence is insufficient.

Variable short-name dictionary:
{short_map_json}

Current triplet batch:
{{{{TRIPLES_JSON_USING_SHORTNAMES}}}}

Domain knowledge:
{self.generate_instructor_scene()}

Missingness report:
{{{{MISSING_REPORT_JSON}}}}

Differential feedback:
{{{{DIFF_FEEDBACK_TEXT}}}}
"""
        return {"prompt": prompt}


class ObjTaskASI(BaseObjTask):
    DATASET_NAME = "ASI"
    SHORT_NAME_MAP = {
        "HF": "Residual_heat_flux",
        "SW": "Residual_shortwave",
        "LW": "Residual_longwave",
        "SLP": "Residual_SLP",
        "TP": "Residual_tot_precip",
        "RH": "Residual_RH",
        "U10": "Residual_u10m",
        "V10": "Residual_v10m",
        "ICE": "Residual_sea_ice",
        "CC": "Residual_cloud_cover",
        "CW": "Residual_cloud_water",
        "GH": "Residual_GH_mean",
    }
    DOMAIN_KNOWLEDGE = (
        "The ASI data describe Arctic sea-ice and atmosphere-ocean coupling after residualization. "
        "Sea ice, radiation, cloud cover, cloud water, heat flux, humidity, precipitation, wind, "
        "sea-level pressure, and geopotential height may interact through short-term anomaly dynamics. "
        "Prefer physically plausible and sparse direct edges. Large-scale circulation variables such as "
        "SLP and GH are often upstream context variables. Cloud and moisture variables can share common "
        "drivers, so avoid dense direct edges among them unless the mechanism is clear."
    )


class ObjTaskASIA(BaseObjTask):
    DATASET_NAME = "ASIA"
    SHORT_NAME_MAP = {
        "A": "asia",
        "S": "smoke",
        "T": "tub",
        "L": "lung",
        "B": "bronc",
        "E": "either",
        "X": "xray",
        "D": "dysp",
    }
    DOMAIN_KNOWLEDGE = (
        "The ASIA network is a classical medical Bayesian network. Risk factors and diseases "
        "should generally point toward diagnostic findings and symptoms, not the reverse."
    )


class ObjTaskChild(BaseObjTask):
    DATASET_NAME = "Child"
    SHORT_NAME_MAP = {}
    DOMAIN_KNOWLEDGE = (
        "The Child network describes infant cardiopulmonary disease, physiological mechanisms, "
        "clinical observations, and reports. Prefer upstream disease or physiology variables pointing "
        "toward downstream observations and reports."
    )


class ObjTaskCRAC(BaseObjTask):
    DATASET_NAME = "CRAC"
    SHORT_NAME_MAP = {}
    DOMAIN_KNOWLEDGE = (
        "The CRAC data describe computer-room air-conditioner supply air temperatures. Prefer sparse, "
        "local, physically plausible edges among nearby units; avoid long-range edges unless strongly justified."
    )


class ObjTaskSAN(BaseObjTask):
    DATASET_NAME = "SAN"
    SHORT_NAME_MAP = {}
    DOMAIN_KNOWLEDGE = (
        "The SAN data describe a Sangiovese viticulture setting. Treatment is an upstream intervention; "
        "growth, canopy, yield, and fruit-quality variables should follow plausible agricultural process order."
    )
