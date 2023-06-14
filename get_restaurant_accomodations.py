import pandas as pd

data = pd.read_csv("_Resto_for_ImageRecog.csv")  # PATH TO CSV

def get_restaurant(prediction, dataset_resto):
    monas = "ChIJLbFk59L1aS4RyLzp4OHWKj0"
    museum_fatahillah = "ChIJfaWSQv8dai4RRQeMZy0D8BI"
    print("prediction: " + prediction)
    if prediction == "monas":
        place_id = monas
    elif prediction == "kotatua":
        place_id = museum_fatahillah

    nearest_resto = dataset_resto[dataset_resto["par_id"] == place_id].sort_values(["popularity", "distance_part_of_cluster"], ascending=[False, False])
    print("nearest_resto =")
    print(nearest_resto)

    top_20 = nearest_resto.sample(n=20)
    selection = top_20.sample(n=7).sort_values("popularity", ascending=False)

    return selection.to_json(orient="records")

hasil_image_recognition = "monas"
rekomendasi_restoran = get_restaurant(prediction=hasil_image_recognition, dataset_resto=data)