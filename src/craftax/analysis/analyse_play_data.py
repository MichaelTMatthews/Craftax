import pickle


def view_play_data():
    with open(
        "/home/mikey/PycharmProjects/Craftax_Internal/play_data/trajectories_1707838447.pkl",
        "rb",
    ) as f:
        x = pickle.load(f)
        print("!")


if __name__ == "__main__":
    view_play_data()
