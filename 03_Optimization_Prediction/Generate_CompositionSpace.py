import pandas as pd
import itertools

elements = ["Fe", "Co", "Ni"]
increments = 1

compositions = []
for values in itertools.product(range(0, 101, increments), repeat=3):
    if sum(values) == 100:
        compositions.append(values)

composition_df = pd.DataFrame(compositions, columns=elements)
composition_df.to_excel("compositions.xlsx", index=False)


