此版本代码集成了：
1. yolo 512
2. seq search
3. 用mask drop object in hand
4. 当pick up mask是空的时候，不在用这个mask拿起东西
5. 建图模块的minz增大了
6. 记录交互失败的位置，交互的时候不去这些位置，相应的fmmplanner里面也有修改

7. 将open object and put object 也改为consecutive interaction

