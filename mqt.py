import paho.mqtt.client as mqtt

class XenderMQTTClient:
    def __init__(self, broker: str = "10.12.54.122"):
        self.client = mqtt.Client()
        self.client.connect(broker)
        self.received_messages = 0
        
    def publish_command(self, command: str):
        """rate"""
        self.client.publish("xender/control", command,qos=1)

    def subscribe(self, topic: str):
        """...."""
        self.client.subscribe(topic,qos=1)
        self.client.on_message = self.on_message

    def on_message(self, client, userdata, msg):
        self.received_messages=int(msg.payload.decode("utf-8"))
        print(f"Received: {msg.topic} {self.received_messages}")