from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/train22/weights/best.pt') 

    # 开始训练
    model.train(
        data='data/datasets.yaml',       # 数据集配置文件
        epochs=300,                # 训练轮数
        batch=32,                  # 批量大小
        imgsz=(640, 800, 1024),    # 启用多尺度训练
        lr0=0.003,                 # 调整学习率
        lrf=0.00001,               # 最低学习率
        optimizer='AdamW',         # 更适合深度学习的优化器
        mosaic=1.0,                # Mosaic 数据增强
        mixup=0.3,                 # Mixup 数据增强
        augment=True,              # 启用数据增强
        freeze=[0,1],                # 冻结部分层，仅训练检测头
        device=0                   # 使用 GPU
    )

    # 验证模型
    metrics = model.val()
    print("验证完成！验证结果：")
    print(metrics)

    # 推理测试
    results = model.predict(source='data/images/test', save=True)
    print("推理测试完成，预测结果已保存！")

    print("训练和验证流程已完成。")

if __name__ == "__main__":
    main()
