# 五任务：学习策略与专家策略视频

每个任务两段720p视频，使用同一初始场景。学习策略重放已记录的Jittor主基线成功动作；专家为MetaWorld规则脚本，不是训练出的神经网络。
片尾1秒是明确标注的定格，不代表学会了保持。20fps为播放速率，不是控制频率。
这些是成功案例演示，不表示所有场景都成功，也不是新一轮评估。
本次在Windows本机重放Linux记录。保存动作与首次成功步数已核对；接触任务可能产生累计奖励差异，详见verification.json，原始基准分数不替换。

|任务|Jittor学习策略|MetaWorld专家|
|---|---|---|
|reach-v2|[64步成功](reach-v2__learned_jittor__seed12000__720p.mp4)|[47步成功](reach-v2__scripted_expert__seed12000__720p.mp4)|
|reach-wall-v2|[48步成功](reach-wall-v2__learned_jittor__seed12000__720p.mp4)|[50步成功](reach-wall-v2__scripted_expert__seed12000__720p.mp4)|
|push-v2|[71步成功](push-v2__learned_jittor__seed12001__720p.mp4)|[66步成功](push-v2__scripted_expert__seed12001__720p.mp4)|
|door-open-v2|[76步成功](door-open-v2__learned_jittor__seed12000__720p.mp4)|[118步成功](door-open-v2__scripted_expert__seed12000__720p.mp4)|
|window-open-v2|[83步成功](window-open-v2__learned_jittor__seed12000__720p.mp4)|[87步成功](window-open-v2__scripted_expert__seed12000__720p.mp4)|