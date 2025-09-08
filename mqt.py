import json
import queue
import threading

import paho.mqtt.client as mqtt

# class XenderMQTTClient:
#     def __init__(self, broker: str = "10.12.54.122"):
#         self.client = mqtt.Client()
#         self.client.connect(broker)
#         self.received_messages = 0
#
#     def publish_command(self, command: str):
#         """rate"""
#         self.client.publish("xender/control", command,qos=1)
#
#     def subscribe(self, topic: str):
#         """...."""
#         self.client.subscribe(topic,qos=1)
#         self.client.on_message = self.on_message
#
#     def on_message(self, client, userdata, msg):
#         self.received_messages=int(msg.payload.decode("utf-8"))
#         print(f"Received: {msg.topic} {self.received_messages}")

adjust_signal = queue.Queue()

class XenderMQTTClient:
    def __init__(self, broker="10.12.54.122"):
        self.received_adjust = False  # 是否收到调整采样率信号
        self.received_done = False    # 是否收到传输完成信号

        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(broker, 1883, 60)
        # 订阅两个主题
        self.client.subscribe("xender/control")
        self.client.subscribe("xender/done")
        self.client.loop_start()

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
            print(f"[Client-MQTT] 收到消息: {msg.topic} -> {payload}")

            if msg.topic == "xender/control" and data.get("action") == "adjust":
                # 收到采样率调整信号
                self.received_adjust = True

            elif msg.topic == "xender/done" and data.get("action") == "done":
                # 收到传输完成信号
                self.received_done = True

        except Exception as e:
            print(f"[Client] 非法消息: {payload}, 错误: {e}")
