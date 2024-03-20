import math


def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * -0.46122 + waga * 0.97314 + wzrost * -0.38203 + 0.80109
    hidden1_po_aktywacji = 1 / (1 + math.exp(-hidden1))
    hidden2 = wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847 + 0.43529
    hidden2_po_aktywacji = 1 / (1 + math.exp(-hidden2))
    output = (
        hidden1_po_aktywacji * -0.81546 + hidden2_po_aktywacji * 1.03775
    ) - 0.2368
    return output


print(forwardPass(23, 75, 176))

people = [
    [23, 75, 176],
    [25, 67, 180],
    [28, 120, 175],
    [22, 65, 165],
    [46, 70, 187],
    [50, 68, 180],
    [48, 97, 178],
]

for person in people:
    print()
    print(person)
    print(forwardPass(*person))
