import pandas as pd; pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
d=pd.read_csv('staking_panel.csv').sort_values(['chain','date'])
g=d.groupby('chain')
print(pd.concat([g.size().rename('n'),g.date.min().rename('d0'),g.tail(1).set_index('chain')[['date','theta','r','E','fee_share']]],axis=1).round(4))
