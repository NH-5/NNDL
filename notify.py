# notify.py
import json
import os
import requests

# 配置文件路径（默认与 notify.py 在同一目录）
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    """
    从 config.json 加载配置。
    如果文件不存在或 JSON 格式错误，会抛出异常。
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"未找到配置文件 {CONFIG_FILE}。\n"
            f"请创建 config.json，并写入：\n"
            f'{{ "BARK_KEY": "你的device_key" }}'
        )
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def bark_send(title, message):
    """
    发送 Bark 推送通知到 iPhone
    :param title: 通知标题
    :param message: 通知内容
    """
    config = load_config()
    bark_key = config.get("BARK_KEY")

    if not bark_key:
        raise ValueError("config.json 中缺少 BARK_KEY 字段！")

    url = f"https://api.day.app/{bark_key}/{title}/{message}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"[Bark] 通知成功: {title} - {message}")
        else:
            print(f"[Bark] 通知失败: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[Bark] 发送通知时发生错误: {e}")
