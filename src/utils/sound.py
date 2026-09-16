import winsound


def alert():
    for _ in range(5):
        winsound.Beep(1000, 500)
        winsound.Beep(2500, 500)
