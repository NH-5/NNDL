import requests
from config import BARK_KEY


def bark_send(title, message):
    """
    发送 Bark 推送通知到 iPhone
    :param title: 通知标题
    :param message: 通知内容
    """

    if not BARK_KEY:
        raise ValueError("缺少 BARK_KEY 字段！")

    url = f"https://api.day.app/{BARK_KEY}/{title}/{message}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"[Bark] 通知成功: {title} - {message}")
        else:
            print(f"[Bark] 通知失败: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[Bark] 发送通知时发生错误: {e}")
