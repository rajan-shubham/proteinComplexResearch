import matplotlib.pyplot as plt
import numpy as np
# state = ['Punjab', 'Harayana', 'UP', 'MP', 'Kerla', 'Rajasthan']
# area = [220, 120, 100, 40, 80, 30]
# plt.bar(state, area, color= 'green', edgecolor= 'yellow')
# plt.xlabel('State')
# plt.ylabel('Area (Lac hectares)')
# plt.title("Wheat Cultivation")
# plt.show()


X = ["a", "b", "c", "d"]
Score = [[5, 25, 45, 20], [4, 23, 49, 17], [6, 22, 47, 19]]
X = np.arange(4)
plt.bar(X+0.00, Score[0], color= "r", width= 0.25)
plt.bar(X+0.25, Score[1], color= "g", width= 0.25)
plt.bar(X-0.25, Score[2], color= "b", width= 0.25)
plt.xticks(X, ["Bumrah", "Rohit", "Dhoni", "Kohli"])
plt.show()