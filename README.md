# Resnet 20
## Lowcost vs Normal
|       Act      |Runtime/epch      |Accuracy       |Flops      |
|----------------|------------------|---------------|-----------|
|Normal          |18s500ms          |93.07@174      |842399744.0|
|Lowcost/4       |                  |91.07@145      |216891392  |
|DW              |                  |89.32@109      |88965120.0 |
|DW2BN           |                  |88.97@122      |88965120.0 |
|Conv->DW2BN->Con|                  |91.70@197      |425394176.0|
|LowcostA        |                  |89.89@122      |425394176.0|
|LowcostAx2      |                  |90.75@121      |425394176.0|
|LowcostA2BN     |                  |90.64@197      |425394176.0|
|LowcostB        |14s603ms          |89.40@120      |425394176.0|
|LowcostC        |15s122ms          |91.88@157      |425394176.0|
|LowcostCrelu6   |15s122ms          |91.44@172      |425394176.0|
|LowcostD        |18s502ms          |91.85@167      |425394176.0|
|LowcostE        |17s702ms          |92.05@170      |425394176.0|
|LowcostCnewlr   |15s122ms          |91.54@176      |425394176.0|

# Graph
## Resnet20 vs Current Best Model
![graph](best.png)