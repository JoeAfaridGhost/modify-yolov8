from ultralytics import YOLO
import onnx
import argparse

def export_model_to_onnx(model_path, export_path, img_size=640):
    """
    将 YOLO 模型导出为 ONNX 格式
    :param model_path: 训练好的 YOLO 模型路径 (.pt 文件)
    :param export_path: 导出 ONNX 文件保存路径
    :param img_size: 模型输入图片尺寸
    """
    print("正在加载模型...")
    model = YOLO(model_path)
    
    print(f"正在导出模型为 ONNX 格式：{export_path}")
    model.export(format='onnx', imgsz=img_size)  # 导出为 ONNX 格式
    print(f"模型已成功导出为 ONNX 格式，保存路径：{export_path}")

    validate_onnx(export_path)

def validate_onnx(onnx_path):
    """
    验证 ONNX 模型
    :param onnx_path: 导出的 ONNX 模型路径
    """
    print(f"正在验证 ONNX 模型：{onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证成功！")
    except Exception as e:
        print(f"ONNX 模型验证失败：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Export to ONNX Script")
    parser.add_argument('--model', type=str, required=True, help="训练好的 YOLO 模型路径 (.pt 文件)")
    parser.add_argument('--output', type=str, required=True, help="ONNX 文件保存路径 (.onnx 文件)")
    parser.add_argument('--imgsz', type=int, default=800, help="模型输入图片尺寸 (默认: 640)")
    args = parser.parse_args()

    export_model_to_onnx(args.model, args.output, args.imgsz)
