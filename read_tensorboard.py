from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 初始化一个EventAccumulator对象
event_acc = EventAccumulator(
    '../tf-logs/2024-06-16-14-19-57/events.out.tfevents.1718518797.autodl-container-ce3c4684af-54aea7eb')

# 加载事件文件
event_acc.Reload()

# 获取所有的标签
tags = event_acc.Tags()

# 获取特定标签的数据
loss_values = event_acc.Scalars('loss')

# 打印每个数据点
for value in loss_values:
    print(value)
