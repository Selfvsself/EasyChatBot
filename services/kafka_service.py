import json

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer


class KafkaService:
    def __init__(self, bootstrap_servers, request_topic, response_topic):
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic

        self.producer: AIOKafkaProducer | None = None
        self.consumer: AIOKafkaConsumer | None = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers
        )
        await self.producer.start()

        self.consumer = AIOKafkaConsumer(
            self.response_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="chat-group"
        )
        await self.consumer.start()

    async def stop(self):
        if self.producer:
            await self.producer.stop()

        if self.consumer:
            await self.consumer.stop()

    async def send_chat_message(self, chat_id, user_id, text, role="assistant"):
        payload = {
            "chat_id": chat_id,
            "user_id": user_id,
            "role": role,
            "text": text
        }

        await self.producer.send_and_wait(
            self.request_topic,
            json.dumps(payload).encode()
        )

    async def send_response_message(self, chat_id, user_id, text, role):
        payload = {
            "chat_id": chat_id,
            "user_id": user_id,
            "role": role,
            "text": text
        }

        await self.producer.send_and_wait(
            self.response_topic,
            json.dumps(payload).encode()
        )
