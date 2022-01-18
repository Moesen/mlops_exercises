import matplotlib.pyplot as plt
import json

data = json.loads(open("times.json", "r").read())
mean = [x["mean"] for x in data]
std = [x["std"] for x in data]

plt.errorbar([x for x in range(1, 13)], mean, std)
plt.title("Runtimes run on AMD Ryzen 7 5800U")
plt.xlabel("Number of workers")
plt.ylabel("Mean time (s)")
plt.xticks([x for x in range(1, 13)])
plt.savefig("load_times.png")