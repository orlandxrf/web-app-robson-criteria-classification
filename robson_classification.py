import json


class RobsonClassification:
    """
    Robson Classification class
    """

    def __init__(self):
        self.groups_descritpion = {}

    def load_robson_group_descriptions(self, json_path: str) -> None:
        """
        Load the 10 Robson Classification group descriptions.

        Args:
            json_path (str). Path to the json file with the 10 Robson Classification group descriptions.

        Returns:
            None
        """
        with open(json_path) as f:
            groups = json.load(f)
        self.groups_descritpion = {int(k): v for k, v in groups.items()}

    def get_group(self, obstetric_entities: dict[str, list[str]]) -> list[str]:
        group = 0

        if "Embarazo múltiple" in obstetric_entities:
            group = 8

        elif "Situación transversa" in obstetric_entities:
            group = 9

        elif "Posición podálica" in obstetric_entities:
            if "Parto multípara" in obstetric_entities:
                group = 7
            else:
                group = 6

        elif "Edad < 37 semanas" in obstetric_entities:
            group = 10

        elif "Parto multípara" in obstetric_entities:
            # Cicatrices uterinas previas
            if "Cesárea previa (Si)" in obstetric_entities:
                group = 5
            elif (
                "TDP inducido" in obstetric_entities
                or "TDP No: cesárea programada" in obstetric_entities
            ):
                group = 4
            else:
                group = 3

        elif (
            "TDP inducido" in obstetric_entities
            or "TDP No: cesárea programada" in obstetric_entities
        ):
            group = 2
        else:
            group = 1

        result = {"group": group, "description": self.groups_descritpion[group]}
        return result
