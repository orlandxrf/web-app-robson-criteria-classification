from datasets import load_from_disk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_ettities_by_note(
    note_id: str,
    tokens: list[str],
    labels: list[str],
):
    # entities = {}
    result = {}
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            entity = labels[i][2:]
            tmp = tokens[i]
            j = i + 1
            # if entity not in entities:
            #     entities[entity] = 1
            # else:
            #     entities[entity] += 1
            while j < len(labels) and labels[j] == "I-" + entity:
                tmp += " " + tokens[j]
                j += 1
            if entity not in result:
                result[entity] = [tmp]
            else:
                result[entity].append(tmp)
            i = j
        elif labels[i].startswith("I-"):
            print(f"\tError: the entity {labels[i]} is not starting with 'B-'")
            print(f"\t{note_id=}")
            print(f"\t{i=}")
            print(f"\t{labels[i]=}")
            exit(0)
        elif labels[i] == "O":
            i += 1
        else:
            print("\tError: something went wrong")
            print(f"\t{note_id=}")
            print(f"\t{i=}")
            print(f"\t{labels[i]=}")
            exit(0)

    # return entities, result
    return result


def robson_classification_diagram(
    note_id: str, obstetrics: dict[str, list[str]]
) -> list[str]:
    groups = []
    # if note_id == "g3-2230-1":
    #     print(f"{note_id=}")
    #     print(f"{obstetrics=}")
    #     exit(0)
    if "Embarazo múltiple" in obstetrics:
        groups.append(8)

    if "Situación transversa" in obstetrics:
        groups.append(9)

    if "Posición podálica" in obstetrics:
        if "Parto multípara" in obstetrics:
            groups.append(7)
        else:
            groups.append(6)

    if "Edad < 37 semanas" in obstetrics:
        groups.append(10)

    if "Parto multípara" in obstetrics:
        # Cicatrices uterinas previas
        if "Cesárea previa (Si)" in obstetrics:
            groups.append(5)
        elif "TDP inducido" in obstetrics or "TDP No: cesárea programada" in obstetrics:
            groups.append(4)
        else:
            groups.append(3)

    if "TDP inducido" in obstetrics or "TDP No: cesárea programada" in obstetrics:
        groups.append(2)

    if len(groups) == 0:
        groups.append(1)

    return groups


def get_parity(entities: dict[str, list[str]]) -> list[any]:
    # "Parto nulípara" --> Nullipara labor, "Parto multípara" --> Multipara labor
    results = []
    if "Parto nulípara" in entities:
        results.append(entities["Parto nulípara"])
    elif "Parto multípara" in entities:
        results.append(entities["Parto multípara"])
    return results


def get_values(obstetrics: list[str], labels: dict[str, list[str]]) -> list[any]:
    results = []
    for label in labels:
        if label in obstetrics:
            results.append(labels[label])
    return None if len(results) == 0 else results


def robson_classification_table(obstetrics: list[str]) -> list[int]:
    result = []
    for i in range(1, 11):
        result.append(
            {
                "group": i,
                "partos": get_values(
                    obstetrics, {"Parto nulípara": "0", "Parto multípara": ">=1"}
                ),
                "cesarea": get_values(
                    obstetrics,
                    {"Cesárea previa (Si)": "Yes", "Cesárea previa (No)": "No"},
                ),
                "fetos": get_values(
                    obstetrics, {"Embarazo único": "1", "Embarazo Múltiple": ">=2"}
                ),
                "presentación": get_values(
                    obstetrics,
                    {
                        "Posición cefálica": "Cefálica",
                        "Posición podálica": "Podalica",
                        "Situación transversa": "Transversa",
                    },
                ),
                "edad": get_values(
                    obstetrics,
                    {"Edad < 37 semanas": "<37", "Edad >= 37 semanas": ">=37"},
                ),
                "tdp": get_values(
                    obstetrics,
                    {
                        "TDP espontáneo": "Espontáneo",
                        "TDP inducido": "Inducido",
                        "TDP No: cesárea programada": "Programado",
                    },
                ),
            }
        )

    return result


if __name__ == "__main__":
    data_name = "gold_notes_80-20"
    dataset = load_from_disk(data_name)
    ner_tags = dataset["train"].features["tags"].feature.names

    obstetric_variables = [
        "Parto nulípara",  # Nullipara labor
        "Parto multípara",  # Multipara labor
        "Cesárea previa (Si)",  # One or more CS
        "Cesárea previa (No)",  # None CS
        "TDP espontáneo",  # Spontaneous labor
        "TDP inducido",  # Induced labor
        "TDP No: cesárea programada",  # No labor, scheduled CS
        "Embarazo único",  # Singleton pregnancy
        "Embarazo múltiple",  # Multiple pregnancy
        "Edad < 37 semanas",  # Preterm pregnancy
        "Edad >= 37 semanas",  # Term pregnancy
        "Posición cefálica",  # Cephalic presentation
        "Posición podálica",  # Breech presentation
        "Situación transversa",  # Transverse lie
    ]

    clasificacion = {
        "note_id": [],
        "grupo": [],
        "tokens": [],
        "tags": [],
    }

    for item in dataset["train"]:
        entities = get_ettities_by_note(
            item["id"], item["tokens"], [ner_tags[t] for t in item["tags"]]
        )
        # print(f"{item['id']}")
        # print(f"{item['tokens']}")
        # print(f"{item['tags']}")
        clasificacion["note_id"].append(item["id"])
        clasificacion["tokens"].append(item["tokens"])
        clasificacion["tags"].append([ner_tags[t] for t in item["tags"]])
        obstetrics = {tg: entities[tg] for tg in entities if tg in obstetric_variables}
        grupo = [0]
        # if item['id'] == "g3-2230-1":
        #     print(f"\t{item['tokens']=}")
        #     print(f"\t{[ner_tags[tg] for tg in item['tags']]=}")
        #     print(f"{obstetrics=}\n")

        if len(obstetrics) > 0:
            # for k, (tk, tg) in enumerate(zip(item['tokens'], item['tags'])):
            #     print(f"{k}\t{tk}\t{ner_tags[tg]}")
            # print(f"\n\t{entities=}")
            # print(f"\t{obstetrics=}")
            # print(f"\tgrupos: {robson_classification_diagram(obstetrics)}")
            # res = robson_classification_table(obstetrics)
            # print(pd.DataFrame(res))
            # print(f"\t{'-'*150}")
            # break
            grupo = robson_classification_diagram(item["id"], obstetrics)
            # print(f"\t{grupo=}")
            # exit(0)
            # if item['id'] == "g3-2230-1":
            #     print(f"\t{item['tokens']=}")
            #     print(f"\t{[ner_tags[tg] for tg in item['tags']]=}")
            #     print(f"{obstetrics=}\n")
            #     print(f"{grupo=}")
            #     exit(0)

        clasificacion["grupo"].append(grupo[0])

    # for i, (tk, lb) in enumerate(zip(dataset["train"][0]["tokens"], dataset["train"][0]["tags"])):
    #     if lb != 0:
    #         lb = ner_tags[lb]
    #         print(f"{i}\t{tk}\t{lb}")

    # print(f"{ner_tags=}")
    # print(dataset["train"][0]["tokens"])
    # print(dataset["train"][0]["tags"])

    df = pd.DataFrame(clasificacion)
    # df.to_csv("sentences/outputs_llama/test_robson.tsv", sep="\t", index=False)

    groups = df["grupo"].value_counts().to_dict()
    for g in range(0, 11):
        if g not in groups:
            groups.update({g: 0})
    groups = dict(sorted(groups.items(), key=lambda x: x[0], reverse=False))
    print(f"{groups=}")

    x = list(groups.keys())
    y = list(groups.values())
    print(f"{x=}")
    print(f"{y=}")

    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    plt.bar(x, y)
    for i in x:
        plt.text(i, y[i], y[i], ha="center")
    plt.title("Clasificación de Criterios de Robson")
    # plt.xlabel("Grupos")
    plt.ylabel("Frecuencia")
    plt.xticks(
        x, tuple([f"grupo {g}" if g != 0 else f"Sin grupo" for g in groups.keys()])
    )
    plt.tight_layout()
    plt.savefig("sentences/img/clasificacion_robson.png", dpi=300, bbox_inches="tight")
    plt.show()
