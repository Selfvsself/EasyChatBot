from abc import abstractmethod


class BaseTool:

    def __init__(self, llm_client, message_repo):
        self.llm = llm_client
        self.message_repo = message_repo


    @abstractmethod
    async def processing(self, chat, app, text):
        pass
