import json

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

class XenderMQTTClient:
    def __init__(self, broker="10.12.54.122"):
        self.received_adjust = False
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(broker, 1883, 60)
        self.client.subscribe("xender/control")
        self.client.loop_start()

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
            if data.get("action") == "adjust":
                # print("[Client] 收到MQTT: 调整采样率")
                self.received_adjust = True
        except:
            print("[Client] 非法消息:", payload)