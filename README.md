# pytorch-template

pytorch使用模板，简化并规范模块编写。

```bash
├── callbacks
│   ├── __init__.py
│   ├── earlystoping.py             # 	早停回调函数
│   ├── lrscheduler.py              # 	学习率调整回调函数
│   ├── modelcheckpoint.py          # 	模型保存回调函数
│   ├── progressbar.py              # 	进度条回调函数
│   ├── writetensorboard.py         # 	tensorboard回调函数
├── config
│   ├── cfg.py                      #   配置类， 配置文件的保存和加载
│   ├── __init__.py 			
│   └── logger.py                   # 	日志类， 打印并保存训练中相关信息以及训练结果
├── data                            #   数据加载模块
│   ├── dataset.py                  # 	借助torch.nn.data加载以及处理数据集
│   └── __init__.py
├── loss                            #   该模块存放自定义损失函数
│   ├── loss.py                     # 	自定义损失函数
│   └── __init__.py
├── model                           #   该模块存放网络模型结构
│   ├── xxxx.py                     # 	网络模型结构
│   └── __init__.py
├── output                          #   该模块存放模型输出  
│   └── xxxxxxxx_xxxx               # 	训练结果 文件夹名为时间
│           ├── checkpoint_dir      # 	模型保存文件夹
│           ├── logs                # 	日志保存文件夹
│           ├── TSboard             # 	tensorboard日志保存文件夹
│           └── config.json         # 	配置文件
├── trainer                         #   该模块存放模型训练器
│   └── trainer.py                  # 	训练器基类
│── utils                           #   通用的函数和类模块
│   ├── metrics .py                 # 	评估函数
│   └── util.py                     # 	工具函数
└── run.py                          #   运行文件   
```
