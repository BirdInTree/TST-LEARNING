import pandas as pd

a = [1,1,1,2,2,2,3,3,3,4,4,4]
b= [3,2,1,4,3,2,5,4,3,6,5,4]
c = [1,2,3,4,5,6,7,8,9,10,11,12]

data = {'unit':a, 'cycle':b, 'value':c}
df = pd.DataFrame(data)
group = df.groupby('unit')
max_cycle = group['cycle'].max()
print(max_cycle)
result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit', right_index=True)
print(result_frame)